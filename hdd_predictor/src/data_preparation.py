import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

# Определение базовых путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PREPARED_DATA_DIR = os.path.join(DATA_DIR, 'prepared')


def prepare_data():
    print("Начало выполнения функции prepare_data_for_training()")

    # Ищем CSV файл в директории RAW_DATA_DIR
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"CSV файл не найден в директории {RAW_DATA_DIR}")

    input_file = csv_files[0]  # Берем первый найденный CSV файл
    input_path = os.path.join(RAW_DATA_DIR, input_file)
    print(f"Найден входной файл: {input_path}")

    # Создаем директорию для выходного файла, если она не существует
    os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
    print(f"Директория PREPARED_DATA_DIR создана: {os.path.exists(PREPARED_DATA_DIR)}")

    output_path = os.path.join(PREPARED_DATA_DIR, 'prepared_dataset.parquet')
    print(f"Путь для выходного файла: {output_path}")

    # Читаем CSV файл
    print("Чтение CSV файла...")
    df = pd.read_csv(input_path)
    print(f"Загружено {len(df)} строк данных.")

    # Анализ значений в столбце failure
    failure_counts = df['failure'].value_counts()
    print("\\nРаспределение значений в столбце failure:")
    print(failure_counts)

    # Преобразуем столбцы даты в datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['serial_number', 'date'])

    # Находим дату первого отказа для каждого диска
    failed_dates = df[df["failure"] == 1].groupby("serial_number")["date"].min().reset_index()
    failed_dates.rename(columns={"date": "failure_date"}, inplace=True)

    # Присоединяем информацию о дате отказа к основному датафрейму
    df = df.merge(failed_dates, on="serial_number", how="left")

    # Удаляем записи после даты отказа и записи без отказа
    df['valid_record'] = df.apply(
        lambda row: row['date'] <= row['failure_date'] if pd.notnull(row['failure_date']) else False, axis=1)
    df = df[df['valid_record']]

    # Вычисляем days_to_failure
    def calculate_days_to_failure(row):
        if pd.isnull(row["failure_date"]):
            return None
        days = (row["failure_date"] - row["date"]).days
        return max(days, 0)

    df["days_to_failure"] = df.apply(calculate_days_to_failure, axis=1)

    # Удаляем ненужные столбцы
    columns_to_keep = [col for col in df.columns if not col.endswith("_raw")]
    df = df[columns_to_keep]

    # Сохраняем обработанные данные в формате Parquet
    df.to_parquet(output_path, index=False)

    # Проверяем, что файл был успешно создан
    if os.path.exists(output_path):
        print(f"Обработанные данные успешно сохранены в {output_path}")
    else:
        print(f"ОШИБКА: Не удалось создать файл {output_path}")

    print("Завершение выполнения функции prepare_data_for_training()")


if __name__ == "__main__":
    prepare_data()
