import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path

# Определение базовых путей
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
FINETUNING_DATA_DIR = DATA_DIR / 'finetuning'
MODEL_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'plots'

# Инициализация путей к файлам
input_file = FINETUNING_DATA_DIR / 'finetuning_data.csv'
model_path = MODEL_DIR / 'lstm_model.h5'
label_encoder_path = MODEL_DIR / 'label_encoder.joblib'
scaler_path = MODEL_DIR / 'standard_scaler.joblib'
feature_columns_path = MODEL_DIR / 'feature_columns.txt'
removed_columns_path = MODEL_DIR / 'removed_columns.txt'
comparison_plot_path = PLOTS_DIR / 'model_comparison_finetuning.png'
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


import argparse
from pathlib import Path




def load_and_prepare_data(file_path, nan_threshold=0.8):
    print("Загрузка и подготовка данных...")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    print(f"Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Исходный размер датафрейма: {df.shape}")

    # Переименование колонок, заканчивающихся на '_raw'
    raw_columns = [col for col in df.columns if col.endswith('_raw')]
    rename_dict = {col: col.replace('_raw', '_normalized') for col in raw_columns}
    df = df.rename(columns=rename_dict)

    # Обработка переименованных колонок
    for old_col, new_col in rename_dict.items():
        if new_col in df.columns and old_col != new_col:
            # Если колонка с '_normalized' уже существует, заменяем данные
            mask = df[new_col].notna()
            df.loc[mask, new_col] = df.loc[mask, old_col]

            # Нормализация перезаписанных данных, сохраняя NaN
            non_nan_mask = df[new_col].notna()
            if non_nan_mask.any():
                mean = df.loc[non_nan_mask, new_col].mean()
                std = df.loc[non_nan_mask, new_col].std()
                df.loc[non_nan_mask, new_col] = (df.loc[non_nan_mask, new_col] - mean) / std

                # Загрузка списка колонок, используемых для предсказания
    with open(feature_columns_path, 'r') as f:
        required_features = [line.strip() for line in f]

        # Добавляем 'model' и 'days_to_failure' к списку необходимых колонок
    required_features += ['model', 'days_to_failure']

    # Проверка наличия всех необходимых колонок
    missing_columns = set(required_features) - set(df.columns)
    if missing_columns:
        error_message = f"В датафрейме отсутствуют следующие необходимые колонки: {', '.join(missing_columns)}"
        print(f"ОШИБКА: {error_message}")
        raise ValueError(error_message)

        # Оставляем только нужные колонки
    df = df[required_features]
    print(f"Размер датафрейма после выбора нужных колонок: {df.shape}")

    # Удаление строк с большим количеством NaN
    df = df.dropna(thresh=len(df.columns) - int(len(df.columns) * nan_threshold))
    print(f"Размер датафрейма после удаления строк с большим количеством NaN: {df.shape}")

    # Обработка выбросов
    print("Обработка выбросов...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[numeric_columns] = df[numeric_columns].clip(lower=lower_bound, upper=upper_bound, axis=1)

    # Заполнение оставшихся NaN значений
    print("Заполнение оставшихся NaN значений...")
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Подготовка данных для модели
    print("Подготовка данных для модели...")
    X_numeric = df.drop(['model', 'days_to_failure'], axis=1)

    # Кодировка категориальной переменной
    print("Кодировка категориальной переменной...")
    le = joblib.load(label_encoder_path)
    X_categorical = le.transform(df['model'])

    y = df['days_to_failure']

    # Нормализация числовых данных
    print("Нормализация числовых данных...")
    scaler = joblib.load(scaler_path)
    X_numeric_scaled = scaler.transform(X_numeric)

    # Преобразование данных в 3D формат
    print("Преобразование данных в 3D формат...")
    X_numeric_3d = X_numeric_scaled.reshape((X_numeric_scaled.shape[0], 1, X_numeric_scaled.shape[1]))

    print(f"Данные подготовлены. Размерность числовых данных: {X_numeric_3d.shape}")
    print(f"Количество категорий: {len(np.unique(X_categorical))}")

    return X_numeric_3d, X_categorical, y

@measure_time
def evaluate_model(model, X_numeric, X_cat, y):
    y_pred = model.predict([X_numeric, X_cat.reshape(-1, 1)]).flatten()

    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean(np.square(y - y_pred)))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    mean_abs_error_days = np.mean(np.abs(y - y_pred))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}%")
    print(f"Модель в среднем ошибается на {mean_abs_error_days:.2f} дней")

    return y_pred, mae, rmse, r2, mape, mean_abs_error_days

def plot_comparison(y_true, y_pred_old, y_pred_new, title):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred_old, alpha=0.5, label='Old Model')
    plt.scatter(y_true, y_pred_new, alpha=0.5, label='New Model')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual days to failure')
    plt.ylabel('Predicted days to failure')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_plot_path)
    plt.close()




@measure_time
def main():
    try:
        parser = argparse.ArgumentParser(description="Retrain the model with new data.")
        parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file")
        args = parser.parse_args()

        input_file = Path(args.input).resolve()  # Преобразуем в абсолютный путь
        print(f"Входной файл: {input_file}")

        if not input_file.exists():
            raise FileNotFoundError(f"Входной файл не найден: {input_file}")

        # Проверка наличия необходимых файлов
        for file_path in [model_path, label_encoder_path, scaler_path, feature_columns_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Загрузка и подготовка данных
        X_numeric_3d, X_categorical, y = load_and_prepare_data(input_file)

        # Разделение данных на обучающую и тестовую выборки
        X_numeric_train, X_numeric_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
            X_numeric_3d, X_categorical, y, test_size=0.2, random_state=42)

        # Загрузка старой модели
        old_model = load_model(model_path, custom_objects={'asymmetric_mse': asymmetric_mse})

        # Оценка старой модели
        print("Оценка старой модели:")
        _, mae_old, rmse_old, r2_old, mape_old, mean_abs_error_days_old = evaluate_model(old_model, X_numeric_test, X_cat_test, y_test)

        # Дообучение модели
        print("Дообучение модели...")
        new_model = load_model(model_path, custom_objects={'asymmetric_mse': asymmetric_mse})
        new_model.fit(
            [X_numeric_train, X_cat_train.reshape(-1, 1)], y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Оценка новой модели
        print("Оценка новой модели:")
        y_pred_new, mae_new, rmse_new, r2_new, mape_new, mean_abs_error_days_new = evaluate_model(new_model, X_numeric_test, X_cat_test, y_test)

        # Сравнение улучшений
        print("\nУлучшения:")
        mae_improvement = (mae_old - mae_new) / mae_old * 100
        rmse_improvement = (rmse_old - rmse_new) / rmse_old * 100
        r2_improvement = (r2_new - r2_old) / r2_old * 100
        mape_improvement = (mape_old - mape_new) / mape_old * 100
        mean_abs_error_days_improvement = (mean_abs_error_days_old - mean_abs_error_days_new) / mean_abs_error_days_old * 100

        print(f"MAE: {mae_improvement:.2f}%")
        print(f"RMSE: {rmse_improvement:.2f}%")
        print(f"R2: {r2_improvement:.2f}%")
        print(f"MAPE: {mape_improvement:.2f}%")
        print(f"Mean Absolute Error Days: {mean_abs_error_days_improvement:.2f}%")

        # Визуальное сравнение
        y_pred_old = old_model.predict([X_numeric_test, X_cat_test.reshape(-1, 1)]).flatten()
        plot_comparison(y_test, y_pred_old, y_pred_new, 'Сравнение предсказаний старой и новой моделей')

        # Определение, является ли новая модель лучше старой
        is_new_model_better = mae_new < mae_old and rmse_new < rmse_old and r2_new > r2_old and mape_new < mape_old

        # Создание директорий для сохранения моделей
        best_model_dir = MODEL_DIR / 'best'
        old_model_dir = MODEL_DIR / 'old'
        best_model_dir.mkdir(exist_ok=True)
        old_model_dir.mkdir(exist_ok=True)

        if is_new_model_better:
            # Сохранение новой модели как лучшей
            new_model_path = best_model_dir / 'lstm_model.h5'
            new_model.save(new_model_path)
            print(f"Новая модель сохранена как лучшая в {new_model_path}")

            # Перемещение старой модели в директорию old
            old_model_path = old_model_dir / 'lstm_model.h5'
            shutil.move(model_path, old_model_path)
            print(f"Старая модель перемещена в {old_model_path}")
        else:
            print("Новая модель не превзошла старую. Сохранение не требуется.")

    except ValueError as e:
        print(f"Ошибка при подготовке данных: {e}")
        return

if __name__ == "__main__":
    main()


