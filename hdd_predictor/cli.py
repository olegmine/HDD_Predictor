#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

# Определение базового пути к директории с исходным кодом
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "hdd_predictor" / "src"

def run_script(script_name, *args):
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        print(f"Ошибка: Скрипт {script_name} не найден.")
        return

    try:
        subprocess.run([sys.executable, str(script_path)] + list(args), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении {script_name}: {e}")

def train_model(input_file):
    print(f"Обучение модели с использованием данных из {input_file}...")
    run_script("train_model.py", input_file)

def retrain_model(input_file):
    print(f"Дообучение модели с использованием данных из {input_file}...")
    run_script("retrain.py", input_file)

def predict(input_file):
    print(f"Предсказание с использованием данных из {input_file}...")
    run_script("prediction.py", input_file)

def model_info():
    print("Получение информации о модели...")
    run_script("model_info.py")

def auto_retrain(interval):
    print(f"Запуск автоматического дообучения с интервалом {interval} секунд...")
    run_script("auto_retrain.py", "--interval", str(interval))

def main():
    parser = argparse.ArgumentParser(description="Консольная утилита для работы с моделью прогнозирования отказов HDD")
    subparsers = parser.add_subparsers(dest="command")

    # Подкоманда обучения модели
    train_parser = subparsers.add_parser("train", help="Обучение модели")
    train_parser.add_argument("input_file", type=str, help="Путь к файлу для обучения")

    # Подкоманда дообучения модели
    retrain_parser = subparsers.add_parser("retrain", help="Дообучение модели")
    retrain_parser.add_argument("input_file", type=str, help="Путь к файлу для дообучения")

    # Подкоманда предсказания
    predict_parser = subparsers.add_parser("predict", help="Предсказание")
    predict_parser.add_argument("input_file", type=str, help="Путь к файлу для предсказания")

    # Подкоманда информации о модели
    model_info_parser = subparsers.add_parser("info", help="Информация о модели")

    # Подкоманда автоматического дообучения
    auto_retrain_parser = subparsers.add_parser("auto_retrain", help="Автоматическое дообучение модели")
    auto_retrain_parser.add_argument("--interval", type=int, default=3600, help="Интервал дообучения в секундах")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.input_file)
    elif args.command == "retrain":
        retrain_model(args.input_file)
    elif args.command == "predict":
        predict(args.input_file)
    elif args.command == "info":
        model_info()
    elif args.command == "auto_retrain":
        auto_retrain(args.interval)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()