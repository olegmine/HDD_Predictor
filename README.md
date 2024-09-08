# HDD Predictor  

HDD Predictor - это инструмент для прогнозирования отказов жестких дисков (HDD) на основе машинного обучения. Проект использует модель LSTM для анализа данных SMART и предсказания вероятности отказа HDD.  

## Особенности  

- Обучение модели на исторических данных SMART  
- Дообучение модели на новых данных  
- Предсказание вероятности отказа HDD  
- Автоматическое периодическое дообучение модели  
- Консольный интерфейс для удобного использования  

## Установка  

1. Клонируйте репозиторий:

- - https://github.com/olegmine/HDD_Predictor/tree/master

2. Установите зависимости:
- - pip install requirements.txt

## Использование  

После установки вы можете использовать следующие команды:  
usage: cli.py [-h] {train,retrain,predict,info,auto_retrain}
1. Обучение модели:
- train 

2. Дообучение модели:
- retrain

3. Предсказание:
- predict

4. Получение информации о модели:
- info

5. Запуск автоматического дообучения:
- auto_retrain

## Структура проекта


- hdd-predictor/
- ├── hdd_predictor/
- │ ├── src/
- │ │ ├── train_model.py
- │ │ ├── retrain.py
- │ │ ├── prediction.py
- │ │ ├── model_info.py
- │ │ └── auto_retrain.py
- │ └── cli.py
- ├── data/
- │ └── finetuning/
- ├── models/
- ├── plots/
- ├── setup.py
- └── README.md


## Требования  

- setuptools~=74.1.2
- pandas~=2.2.2
- numpy~=1.26.4
- joblib~=1.4.2
- matplotlib~=3.9.2
- tensorflow~=2.17.0
- scikit-learn~=1.5.1
- tqdm~=4.66.5


## Авторы  

Jesters

