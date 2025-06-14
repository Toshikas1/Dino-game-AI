# Google Dino AI Evolution

Это проект симуляции нейросетевой эволюции ИИ-динозавров, обучающихся играть в игру Google Dino. ИИ управляет прыжками и приседаниями динозавра, избегая препятствий.

## Особенности

- Эволюционный алгоритм на основе отбора, скрещивания и мутаций
- Нейросеть принимает решения на основе 5 параметров препятствия
- Чекпоинты для сохранения прогресса обучения каждые N поколений
- Визуализация с помощью Pygame

## Установка

1. Установите Python 3.10+
2. Установите зависимости:
    ```bash
    pip install pygame
    ```

## Запуск

```bash
python main.py
```

В начале можно выбрать сохранённый чекпоинт или начать новую игру.

## Структура проекта

- `main.py` — основной игровой цикл
- `generations/` — папка для сохранения чекпоинтов `.pkl`
- `NeuralNetwork` — класс простой нейросети с двумя выходами: прыжок и приседание
- `DinoAI` — управляемый ИИ динозавр
- `Population` — класс популяции и логика эволюции

## Управление

Управление осуществляется автоматически нейросетью. Вы можете наблюдать за процессом обучения.

## Лицензия

Этот проект предоставляется "как есть" в образовательных целях.