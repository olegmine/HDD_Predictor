import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path


def process_prepared_data():
    print("Начало выполнения функции process_prepared_data()")

    # Определение путей
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    PREPARED_DATA_DIR = DATA_DIR / 'prepared'

    input_path = PREPARED_DATA_DIR / 'prepared_dataset.parquet'
    output_path = PREPARED_DATA_DIR / 'combined_dataset_processed.parquet'

    # Проверка существования входного файла
    if not input_path.exists():
        print(f"Содержимое директории {PREPARED_DATA_DIR}:")
        for file in PREPARED_DATA_DIR.iterdir():
            print(f"  {file.name}")
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    # Загружаем данные
    print(f"Загрузка данных из {input_path}...")
    df = pd.read_parquet(input_path)

    # Преобразуем столбцы даты в datetime (если это еще не сделано)
    df['date'] = pd.to_datetime(df['date'])

    # Сортируем DataFrame по serial_number и date
    df = df.sort_values(['serial_number', 'date'])

    # Находим дату первого отказа для каждого диска (если это еще не сделано)
    if 'failure_date' not in df.columns:
        print("Вычисление дат отказа...")
        failed_dates = df[df["failure"] == 1].groupby("serial_number")["date"].min().reset_index()
        failed_dates.rename(columns={"date": "failure_date"}, inplace=True)
        df = df.merge(failed_dates, on="serial_number", how="left")

    # Удаляем записи после даты отказа (если это еще не сделано)
    if 'valid_record' not in df.columns:
        print("Удаление записей после даты отказа...")
        df['valid_record'] = df.apply(
            lambda row: row['date'] <= row['failure_date'] if pd.notnull(row['failure_date']) else True, axis=1)
        df = df[df['valid_record']]

    # Вычисляем days_to_failure (если это еще не сделано)
    if 'days_to_failure' not in df.columns:
        print("Вычисление дней до отказа...")
        tqdm.pandas(desc="Вычисление дней до отказа")
        df["days_to_failure"] = df.progress_apply(
            lambda row: (row["failure_date"] - row["date"]).days if pd.notnull(row["failure_date"]) else None,
            axis=1
        )

    # Проверка на отрицательные значения
    negative_days = df[df["days_to_failure"] < 0]
    if len(negative_days) > 0:
        print(f"\nВНИМАНИЕ: Найдено {len(negative_days)} записей с отрицательным days_to_failure.")
        print("Пример таких записей:")
        print(negative_days[["serial_number", "date", "failure_date", "days_to_failure"]].head())
    else:
        print("\nОтрицательных значений days_to_failure не обнаружено.")

    # Удаляем ненужные столбцы
    columns_to_keep = [col for col in df.columns if not col.endswith("_raw")]
    columns_to_keep = [col for col in columns_to_keep if col not in ["failure_date", "failure", "valid_record"]]
    df = df[columns_to_keep]

    # Показываем результат
    print("\nИтоговая статистика:")
    print("Количество строк после обработки:", len(df))
    print("Количество уникальных серийных номеров:", df["serial_number"].nunique())
    print("\nПервые несколько строк обработанного DataFrame:")
    print(df.head())

    # Сохраняем обработанные данные
    print(f"\nСохранение обработанных данных в {output_path}...")
    df.to_parquet(output_path, index=False)

    print(f"Обработанные данные сохранены в {output_path}")
    print("Завершение выполнения функции process_prepared_data()")


if __name__ == "__main__":
    process_prepared_data()


