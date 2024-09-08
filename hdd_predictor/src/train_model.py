import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Masking, Embedding, Concatenate, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import tensorflow as tf
import time
from pathlib import Path

# Определение базовых путей
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PREPARED_DATA_DIR = DATA_DIR / 'prepared'
MODEL_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'plots'

# Инициализация всех путей к файлам
input_file = PREPARED_DATA_DIR / 'combined_dataset_processed.parquet'
model_path = MODEL_DIR / 'lstm_model.h5'
label_encoder_path = MODEL_DIR / 'label_encoder.joblib'
scaler_path = MODEL_DIR / 'standard_scaler.joblib'
feature_columns_path = MODEL_DIR / 'feature_columns.txt'
actual_vs_predicted_plot_path = PLOTS_DIR / 'actual_vs_predicted.png'
training_loss_plot_path = PLOTS_DIR / 'training_loss.png'
training_mae_plot_path = PLOTS_DIR / 'training_mae.png'
removed_columns_path = MODEL_DIR / 'removed_columns.txt'

# Создание необходимых директорий
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} выполнилась за {end_time - start_time:.2f} секунд")
        return result
    return wrapper

def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    underestimation_penalty = 2.0
    return tf.where(error > 0, tf.square(error), tf.square(error) * underestimation_penalty)

@measure_time
def main():
    print("Начало обработки данных и обучения модели...")

    # Загрузка данных
    print(f"Загрузка данных из {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Исходный размер датафрейма: {df.shape}")

    # Предобработка данных
    nan_threshold = 0.8
    print("Удаление колонок с большим количеством NaN...")
    nan_columns = df.columns[df.isnull().mean() > nan_threshold].tolist()
    df = df.drop(columns=nan_columns)
    print(f"Размер датафрейма после удаления NaN колонок: {df.shape}")

    with open(removed_columns_path, 'w') as f:
        f.write('\n'.join(nan_columns))

    print("Обработка выбросов...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[numeric_columns] = df[numeric_columns].clip(lower=lower_bound, upper=upper_bound, axis=1)

    print("Заполнение оставшихся NaN значений...")
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Подготовка данных для модели
    print("Подготовка данных для модели...")
    feature_columns = df.select_dtypes(include=[np.number]).columns.drop(['days_to_failure'])
    X_numeric = df[feature_columns]

    print("Кодировка категориальной переменной...")
    le = LabelEncoder()
    X_categorical = le.fit_transform(df['model'])

    y = df['days_to_failure']

    print("Нормализация числовых данных...")
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    print("Преобразование данных в 3D формат...")
    X_numeric_3d = X_numeric_scaled.reshape((X_numeric_scaled.shape[0], 1, X_numeric_scaled.shape[1]))

    # Разделение данных на обучающую и тестовую выборки
    X_numeric_train, X_numeric_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_numeric_3d, X_categorical, y, test_size=0.2, random_state=42)

    # Создание и компиляция модели
    print("Создание и компиляция модели...")
    model = create_and_compile_model((1, X_numeric_3d.shape[2]), len(le.classes_),
                                     dropout_rate=0.2, l1_reg=0.01, l2_reg=0.01, use_batch_norm=True)

    # Обучение модели
    print("Обучение модели...")
    history = train_model(model, X_numeric_train, X_cat_train, y_train)

    # Сохранение модели и вспомогательных объектов
    model.save(model_path)
    joblib.dump(le, label_encoder_path)
    joblib.dump(scaler, scaler_path)
    with open(feature_columns_path, 'w') as f:
        for column in feature_columns:
            f.write(f"{column}\n")
    print(f"Модель и вспомогательные объекты сохранены в {MODEL_DIR}")

    # Оценка модели
    print("Оценка модели...")
    y_pred, test_mae, test_rmse, r2, mape, mean_abs_error_days = evaluate_model(model, X_numeric_test, X_cat_test, y_test)

    # Построение графиков
    print("Построение графиков...")
    plot_results(y_test, y_pred, history)

    print("Процесс завершен. Все файлы сохранены.")

def create_and_compile_model(input_shape, num_categories, dropout_rate=0.2, l1_reg=0.01, l2_reg=0.01, use_batch_norm=True):
    numeric_input = Input(shape=input_shape)
    masked_numeric = Masking(mask_value=np.nan)(numeric_input)

    lstm_out_large = LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                          recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                          return_sequences=True)(masked_numeric)
    if use_batch_norm:
        lstm_out_large = BatchNormalization()(lstm_out_large)
    lstm_out_large = Dropout(dropout_rate)(lstm_out_large)

    lstm_out1 = LSTM(75, activation='tanh', recurrent_activation='sigmoid',
                     kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                     recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                     return_sequences=True)(lstm_out_large)
    if use_batch_norm:
        lstm_out1 = BatchNormalization()(lstm_out1)
    lstm_out1 = Dropout(dropout_rate)(lstm_out1)

    lstm_out2 = LSTM(40, activation='tanh', recurrent_activation='sigmoid',
                     kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                     recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(lstm_out1)
    if use_batch_norm:
        lstm_out2 = BatchNormalization()(lstm_out2)
    lstm_out2 = Dropout(dropout_rate)(lstm_out2)

    categorical_input = Input(shape=(1,))
    embedding = Embedding(input_dim=num_categories, output_dim=8)(categorical_input)
    embedding = Flatten()(embedding)

    merged = Concatenate()([lstm_out2, embedding])

    dense1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(merged)
    if use_batch_norm:
        dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(dense1)
    if use_batch_norm:
        dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(dropout_rate)(dense2)

    output = Dense(1)(dense2)

    model = Model(inputs=[numeric_input, categorical_input], outputs=output)

    optimizer = Adam(learning_rate=0.003, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=asymmetric_mse, metrics=['mae', 'mse'])

    return model

def train_model(model, X_numeric_train, X_cat_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        [X_numeric_train, X_cat_train.reshape(-1, 1)], y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return history

def evaluate_model(model, X_numeric_test, X_cat_test, y_test):
    test_loss, test_mae, test_mse = model.evaluate([X_numeric_test, X_cat_test.reshape(-1, 1)], y_test, verbose=0)
    test_rmse = np.sqrt(test_mse)

    y_pred = model.predict([X_numeric_test, X_cat_test.reshape(-1, 1)]).flatten()

    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mean_abs_error_days = np.mean(np.abs(y_test - y_pred))

    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}%")
    print(f"Модель в среднем ошибается на {mean_abs_error_days:.2f} дней")

    return y_pred, test_mae, test_rmse, r2, mape, mean_abs_error_days

def plot_results(y_test, y_pred, history):
    # График "Actual vs Predicted Days to Failure"
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='gray')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Фактические дни до отказа')
    plt.ylabel('Предсказанные дни до отказа')
    plt.title('Фактические vs Предсказанные дни до отказа')
    plt.tight_layout()
    plt.savefig(actual_vs_predicted_plot_path)
    plt.close()

    # График "Model Loss"
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Ошибка на обучающей выборке', color='gray')
    plt.plot(history.history['val_loss'], label='Ошибка на валидационной выборке', color='black')
    plt.title('Ошибка модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_loss_plot_path)
    plt.close()

    # График "Model MAE"
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='MAE на обучающей выборке', color='gray')
    plt.plot(history.history['val_mae'], label='MAE на валидационной выборке', color='black')
    plt.title('Средняя абсолютная ошибка модели')
    plt.xlabel('Эпоха')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_mae_plot_path)
    plt.close()

if __name__ == "__main__":
    main()