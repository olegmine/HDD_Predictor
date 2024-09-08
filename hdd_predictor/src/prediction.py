import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import argparse

# Определение базовых путей
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
PREPARED_DATA_DIR = DATA_DIR / 'prepared'

# Инициализация всех путей к файлам
model_path = MODEL_DIR / 'lstm_model.h5'
label_encoder_path = MODEL_DIR / 'label_encoder.joblib'
scaler_path = MODEL_DIR / 'standard_scaler.joblib'
feature_columns_path = MODEL_DIR / 'feature_columns.txt'

# Статический путь к входному файлу (замените на ваш путь)
STATIC_INPUT_FILE = DATA_DIR / 'input_data.csv'


def load_model_and_preprocessors():
    # Загрузка модели
    model = load_model(model_path, compile=False)

    # Загрузка LabelEncoder
    le = joblib.load(label_encoder_path)

    # Загрузка StandardScaler
    scaler = joblib.load(scaler_path)

    # Загрузка списка признаков
    with open(feature_columns_path, 'r') as f:
        feature_columns = [line.strip() for line in f]

    return model, le, scaler, feature_columns


def preprocess_data(df, le, scaler, feature_columns, nan_threshold=0.2):
    print("Загрузка и подготовка данных...")
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

    # Добавляем 'model' к списку необходимых колонок
    required_features = feature_columns + ['model']

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
    X_numeric = df.drop(['model'], axis=1)

    # Кодировка категориальной переменной
    print("Кодировка категориальной переменной...")
    X_categorical = le.transform(df['model'])

    # Нормализация числовых данных
    print("Нормализация числовых данных...")
    X_numeric_scaled = scaler.transform(X_numeric)

    # Преобразование данных в 3D формат
    print("Преобразование данных в 3D формат...")
    X_numeric_3d = X_numeric_scaled.reshape((X_numeric_scaled.shape[0], 1, X_numeric_scaled.shape[1]))

    print(f"Данные подготовлены. Размерность числовых данных: {X_numeric_3d.shape}")
    print(f"Количество категорий: {len(np.unique(X_categorical))}")

    return X_numeric_3d, X_categorical


def make_predictions(model, X_numeric, X_categorical):
    return model.predict([X_numeric, X_categorical.reshape(-1, 1)]).flatten()


def main(input_file):
    # Загрузка модели и препроцессоров
    model, le, scaler, feature_columns = load_model_and_preprocessors()

    # Загрузка данных
    print(f"Загрузка данных из {input_file}...")
    df = pd.read_csv(input_file)

    # Предобработка данных
    X_numeric, X_categorical = preprocess_data(df, le, scaler, feature_columns)

    # Получение предсказаний
    predictions = make_predictions(model, X_numeric, X_categorical)

    # Добавление предсказаний в исходный датафрейм
    df['predicted_days_to_failure'] = predictions

    # Убедитесь, что директория для сохранения результатов существует
    output_dir = Path(input_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение результатов
    output_file = output_dir / f"{Path(input_file).stem}_with_predictions.csv"
    df.to_csv(output_file, index=False)

    print(f"Предсказания сохранены в файл: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict days to failure for hard drives")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    if args.input_file:
        input_file_path = Path(args.input_file)
    else:
        input_file_path = STATIC_INPUT_FILE

    if not input_file_path.exists():
        print(f"Файл {input_file_path} не существует.")
    else:
        main(input_file_path)




