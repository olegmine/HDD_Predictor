import subprocess
import sys
from pathlib import Path
import time
import argparse

# Определение базовых путей
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
DATA_DIR = BASE_DIR / 'data'
FINETUNING_DIR = DATA_DIR / 'finetuning'
PREPARED_DIR = DATA_DIR / 'prepared'

# Убедимся, что директории существуют
FINETUNING_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

# Пути к скриптам
DATA_PREPARATION_SCRIPT = SRC_DIR / 'data_preparation.py'
DATA_PROCESSING_SCRIPT = SRC_DIR / 'data_processing.py'
RETRAIN_SCRIPT = SRC_DIR / 'retrain.py'

# Пути к файлам данных
FINETUNING_DATA_PATH_CSV = FINETUNING_DIR / 'finetuning_data.csv'
FINETUNING_DATA_PATH_PARQUET = PREPARED_DIR / 'combined_dataset_processed.parquet'

# Получаем путь к текущему интерпретатору Python
PYTHON_EXECUTABLE = sys.executable


def run_script(script_path, *args):
    """
    Запускает Python скрипт с заданными аргументами.
    """
    command = [PYTHON_EXECUTABLE, str(script_path)] + list(args)
    print(f"Выполняется команда: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"Скрипт {script_path.name} успешно выполнен.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении скрипта {script_path.name}: {e}")
        raise


def convert_parquet_to_csv(parquet_path, csv_path):
    """
    Конвертирует файл .parquet в .csv
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"Данные конвертированы из {parquet_path} в {csv_path}")
    print(f"Размер DataFrame: {df.shape}")
    print(f"Первые несколько строк:\n{df.head()}")


def main(interval):
    while True:
        print(f"Шаг 1: Сбор данных")
        run_script(DATA_PREPARATION_SCRIPT)

        print(f"\nШаг 2: Обработка данных")
        run_script(DATA_PROCESSING_SCRIPT)

        # Проверка, что файл .parquet создан
        if not FINETUNING_DATA_PATH_PARQUET.exists():
            raise FileNotFoundError(f"Файл {FINETUNING_DATA_PATH_PARQUET} не был создан.")

        print("\nКонвертация данных из .parquet в .csv")
        convert_parquet_to_csv(FINETUNING_DATA_PATH_PARQUET, FINETUNING_DATA_PATH_CSV)

        if not FINETUNING_DATA_PATH_CSV.exists():
            raise FileNotFoundError(f"Файл {FINETUNING_DATA_PATH_CSV} не был создан.")

        print(f"Проверка наличия файла: {FINETUNING_DATA_PATH_CSV}")
        if FINETUNING_DATA_PATH_CSV.exists():
            print(f"Файл существует. Размер: {FINETUNING_DATA_PATH_CSV.stat().st_size} байт")
            print(f"Абсолютный путь к файлу: {FINETUNING_DATA_PATH_CSV.resolve()}")
        else:
            print("ОШИБКА: Файл не найден!")
            return

        print("\nШаг 3: Переобучение сети")
        run_script(RETRAIN_SCRIPT, '--input', str(FINETUNING_DATA_PATH_CSV.resolve()))

        print("\nПроцесс автоматического переобучения завершен.")
        print(f"Ожидание {interval} секунд перед следующим запуском...")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Настройка автоматического дообучения модели")
    parser.add_argument("--interval", type=int, default=3600, help="Интервал дообучения в секундах")
    args = parser.parse_args()

    main(args.interval)
