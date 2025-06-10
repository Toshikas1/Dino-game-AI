import pygame
import random
import sys
import math
import pickle
import os
from datetime import datetime

# Константы
MAX_GENERATIONS = 150          # Максимальное количество поколений
CHECKPOINT_EVERY = 5           # Частота сохранения чекпоинтов
CHECKPOINT_DIR = "generations" # Директория для сохранения чекпоинтов
COUNT_PARAMS = 5               # Количество параметров в нейросети
is_good = False                # Флаг для оценки действий ИИ

# Создаем директорию для чекпоинтов, если ее нет
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

class NeuralNetwork:
    """Класс нейронной сети для принятия решений динозавром"""
    
    def __init__(self):
        """Инициализация нейросети со случайными весами"""
        self.jump_weights = []  # Веса для решения о прыжке
        self.duck_weights = []  # Веса для решения о приседании
        self.weights = [self.jump_weights, self.duck_weights]
        
        # Инициализация случайных весов
        for i in range(COUNT_PARAMS):
            self.jump_weights.append(random.uniform(-1, 1)) 
        for i in range(COUNT_PARAMS):
            self.duck_weights.append(random.uniform(-1, 1)) 
            
        self.biases = [random.uniform(-1, 1), random.uniform(-1, 1)]  # Смещения

    def predict(self, inputs: list[float]) -> list[float]:
        """Предсказание действий на основе входных данных
        Args:
            inputs (list[float]): Входные параметры (расстояние, скорость и т.д.)
        Returns:
            list[float]: Вероятности прыжка и приседания
        """
        prediction = 0
        prediction2 = 0
        for i in range(len(inputs)):
            prediction += inputs[i] * self.jump_weights[i]
        for g in range(len(inputs)):
            prediction2 += inputs[i] * self.duck_weights[i]
        prediction += self.biases[0]
        prediction2 += self.biases[1]
        return [sigmoid(prediction), sigmoid(prediction2)]

    def mutate(self) -> None:
        """Мутация весов с заданной вероятностью"""
        for i in range(0, len(self.jump_weights)):
            if random.random() <= MUTATION_RATE:
                self.jump_weights[i] += random.uniform(-0.5, 0.5)
        if random.random() <= MUTATION_RATE:
            self.biases[0] += random.uniform(-0.5, 0.5)
        for i in range(0, len(self.duck_weights)):
            if random.random() <= MUTATION_RATE:
                self.duck_weights[i] += random.uniform(-0.5, 0.5)
        if random.random() <= MUTATION_RATE:
            self.biases[1] += random.uniform(-0.5, 0.5)

    def copy(self) -> 'NeuralNetwork':
        """Создание копии нейросети
        Returns:
            NeuralNetwork: Новая нейросеть с теми же параметрами
        """
        child = NeuralNetwork()
        child.jump_weights, child.duck_weights = self.jump_weights, self.duck_weights
        child.weights = self.weights
        child.biases = self.biases
        return child
        
    def crossover(self, papa: 'NeuralNetwork') -> 'NeuralNetwork':
        """Скрещивание с другой нейросетью
        Args:
            papa (NeuralNetwork): Вторая нейросеть для скрещивания
        Returns:
            NeuralNetwork: Новая нейросеть-потомок
        """
        child_jump_weights = []
        child_duck_weights = []
        for i in range(0, COUNT_PARAMS):
            if random.random() > 0.5:
                child_jump_weights.append(papa.jump_weights[i])
            else:
                child_jump_weights.append(self.jump_weights[i])
            if random.random() > 0.5:
                child_duck_weights.append(papa.duck_weights[i])
            else:
                child_duck_weights.append(self.duck_weights[i])
        if random.random() > 0.5:
            child_biases = papa.biases
        else:
            child_biases = self.biases
        child = NeuralNetwork()
        child.jump_weights = child_jump_weights
        child.duck_weights = child_duck_weights
        child.biases = child_biases
        return child


class DinoAI:
    """Класс динозавра с ИИ"""
    
    def __init__(self):
        """Инициализация динозавра"""
        self.y = ground_y              # Позиция по Y
        self.vel_y = 0                 # Скорость по Y
        self.is_jumping = False        # Флаг прыжка
        self.is_ducking = False       # Флаг приседания
        self.alive = True              # Флаг жизни
        self.brain = NeuralNetwork()   # Нейросеть для принятия решений
        self.fitness = 0               # Приспособленность
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))  # Случайный цвет

    def update(self, cactus: dict) -> None:
        """Обновление состояния динозавра
        Args:
            cactus (dict): Информация о ближайшем кактусе/птице
        """
        if not self.alive:
            return
        self.is_ducking = False

        if cactus:
            # Нормализация входных параметров
            distance = (cactus['x'] - dino_x) / WIDTH
            speed_norm = cactus_speed / MAX_SPEED
            top = 1.0 - (cactus['y'] / HEIGHT)
            bottom = (cactus['y'] + cactus['height']) / HEIGHT
            obstacle_type = 1.0 if cactus['type'] == 'bird' else 0.0
            inputs = [distance, speed_norm, top, bottom, obstacle_type]
            
            # Принятие решения о приседании
            if self.brain.predict(inputs)[1] > 0.5 and not self.is_ducking and not self.is_jumping:
                self.is_ducking = True
                is_good = False
                for cactus in cacti:
                    if isinstance(cactus, list):
                        for el in cactus:
                            if el["x"] <= dino_x + 20 and el["type"] == "bird":
                                is_good = True
                    else:
                        if cactus["x"] <= dino_x + 20 and cactus["type"] == "bird":
                            is_good = True
                        
                # Штраф за неправильное приседание
                if not is_good and population.generation>= 10:
                    self.fitness -= 1
            else:
                self.is_ducking = False

            # Принятие решения о прыжке
            if self.brain.predict(inputs)[0] > 0.5 and not self.is_jumping and not self.is_ducking:
                self.is_jumping = True
                self.vel_y = -jump_height
                is_good=False
                for cactus in cacti:
                    if cactus["x"] <= dino_x + 20 + (cactus_speed * 7) and cactus["type"] == "cactus":
                        is_good = True

                # Штраф за неправильный прыжок
                if not is_good and population.generation>= 10:
                    self.fitness -= 1

        # Физика прыжка
        if self.is_jumping:
            self.y += self.vel_y
            self.vel_y += gravity
            if self.y >= ground_y:
                self.y = ground_y
                self.is_jumping = False

        self.fitness += 1  # Награда за выживание

    def draw(self, surface: pygame.Surface) -> None:
        """Отрисовка динозавра
        Args:
            surface (pygame.Surface): Поверхность для отрисовки
        """
        if self.alive:
            height = duck_height if self.is_ducking else dino_height
            pygame.draw.rect(surface, self.color, (dino_x, self.y+(dino_height-height), dino_width, height))


class Population:
    """Класс популяции динозавров"""
    
    def sort(self) -> None:
        """Сортировка динозавров по приспособленности"""
        for i in range (0, len(self.dinos)):
            for g in range (0, len(self.dinos) - 1 - i):
                if self.dinos[g].fitness < self.dinos[g +1].fitness:
                   self.dinos[g], self.dinos[g+1] = self.dinos[g + 1], self.dinos[g]

    def __init__(self):
        """Инициализация популяции"""
        self.generation = 1            # Текущее поколение
        self.dinos = [DinoAI() for _ in range(POPULATION_SIZE)]  # Список динозавров

    def all_dead(self) -> bool:
        """Проверка, все ли динозавры погибли
        Returns:
            bool: True если все мертвы
        """
        return all(not d.alive for d in self.dinos)

    def evolve(self) -> None:
        """Эволюция популяции (отбор, скрещивание, мутация)"""
        best_fitness = max(d.fitness for d in self.dinos)
        avg_fitness = sum(d.fitness for d in self.dinos) / len(self.dinos)
        print(f"Gen {self.generation} | Best Life: {best_fitness} | Avg Life: {round(avg_fitness)}")

        # Сохранение чекпоинта
        if self.generation % CHECKPOINT_EVERY == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{CHECKPOINT_DIR}/gen_{self.generation}_{timestamp}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self, f)

        if self.generation >= MAX_GENERATIONS:
            print("Max generations reached.")
            pygame.quit()
            sys.exit()

        self.generation += 1
        self.sort()
        best_dinos = self.dinos[:POPULATION_SIZE//5]  # Отбор лучших 20%
        
        # Создание нового поколения
        population = []
        for i in range(0, POPULATION_SIZE):
            parent = random.choice(best_dinos)
            child = DinoAI()
            if i <= POPULATION_SIZE//2:
                child.brain = parent.brain.copy()  # Клонирование
            else:
                child.brain = parent.brain.crossover(random.choice(best_dinos).brain)  # Скрещивание

            child.brain.mutate()  # Мутация
            population.append(child)
            
        self.dinos = population


def sigmoid(x: float) -> float:
    """Сигмоидная функция активации
    Args:
        x (float): Входное значение
    Returns:
        float: Значение в диапазоне (0, 1)
    """
    return 1 / (1 + math.exp(-x))


def spawn_cactus() -> dict | list[dict]:
    """Создание нового препятствия
    Returns:
        dict or list[dict]: Одно препятствие или группа
    """
    global obstacles_passed

    def create_obstacle(force_type=None) -> dict:
        """Создание одного препятствия
        Args:
            force_type (str, optional): Принудительный тип препятствия. Defaults to None.
        Returns:
            dict: Параметры препятствия
        """
        is_bird = force_type == 'bird' or (force_type is None and random.random() < 0.3 and obstacles_passed >= 5)
        if is_bird:
            return {
                'x': WIDTH + random.randint(0, 100),
                'width': 30,
                'height': 30,
                'y': ground_y - 25,
                'type': 'bird',
                'passed': False
            }
        else:
            height = random.randint(25, 35)
            return {
                'x': WIDTH + random.randint(0, 100),
                'width': random.randint(15, 30),
                'height': height,
                'y': HEIGHT - height - 20,
                'type': 'cactus',
                'passed': False
            }

    # Создание группы препятствий после 10 пройденных
    if obstacles_passed >= 10 and random.random() < 0.4:
        num = random.choice([2, 3])
        group = []
        spacing = dino_width * 2 + 10

        start_x = WIDTH + random.randint(0, 100)
        for i in range(num):
            kind = random.choice(['bird', 'cactus'])
            obs = create_obstacle(force_type=kind)
            obs['x'] = start_x + i * spacing + i * 60
            group.append(obs)
        return group

    return create_obstacle()


def draw(cacti: list, obstacles_passed: int, max_obstacles: int = 0) -> None:
    """Отрисовка игры
    Args:
        cacti (list): Список препятствий
        obstacles_passed (int): Количество пройденных препятствий
        max_obstacles (int, optional): Максимальное количество препятствий. Defaults to 0.
    """
    win.fill(WHITE)
    for cactus in cacti:
        if isinstance(cactus, list):
            for el in cactus:
                color = (0, 100, 200) if el['type'] == 'bird' else GREEN
                pygame.draw.rect(win, color, (el['x'], el['y'], el['width'], el['height']))
        else:
            color = (0, 100, 200) if cactus['type'] == 'bird' else GREEN
            pygame.draw.rect(win, color, (cactus['x'], cactus['y'], cactus['width'], cactus['height']))
    
    pygame.draw.line(win, BLACK, (0, HEIGHT - 20), (WIDTH, HEIGHT - 20), 2)
    
    for dino in population.dinos:
        dino.draw(win)
        
    text = font.render(f"Gen: {population.generation} | Alive: {sum(d.alive for d in population.dinos)} | Obstacles: {obstacles_passed} | Max: {max_obstacles} | Speed: {cactus_speed}" , True, BLACK)
    win.blit(text, (10, 10))
    pygame.display.update()


def choose_checkpoint() -> Population | None:
    """Выбор чекпоинта для загрузки
    Returns:
        Population or None: Загруженная популяция или None
    """
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pkl")])
    if not files:
        print("\nНет сохранённых чекпоинтов. Запуск новой игры.")
        return None
        
    print("\nВыберите чекпоинт:")
    for i, file in enumerate(files):
        print(f"{i}: {file}")
        
    while True:
        choice = input("Введите номер чекпоинта или 'n' для новой игры: ").strip()
        if choice.lower() == 'n':
            return None
        if choice.isdigit() and 0 <= int(choice) < len(files):
            filename = os.path.join(CHECKPOINT_DIR, files[int(choice)])
            with open(filename, "rb") as f:
                print(f"Загружается: {files[int(choice)]}")
                return pickle.load(f)
        print("Неверный ввод. Попробуйте снова.")


def main() -> None:
    """Главная функция игры"""
    global cactus_timer, cacti, obstacles_passed, cactus_speed, population
    
    # Инициализация игры
    population = choose_checkpoint() or Population()
    max_obstacles_passed = 0
    run = True
    time_alive = 0
    new_obs = spawn_cactus()
    
    # Главный игровой цикл
    while run:
        clock.tick(60)
        time_alive += 1
        cactus_speed = min(MAX_SPEED, 5 + time_alive // 300)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Логика появления препятствий
        cactus_timer -= 1
        if cactus_timer <= 0 and new_obs is None:
            new_obs = spawn_cactus()

        if new_obs is not None:
            if isinstance(new_obs, list):
                cacti.extend(new_obs)
            else:
                cacti.append(new_obs)
            new_obs = None
            cactus_timer = random.randint(60, 120)

        # Обновление позиций препятствий
        for cactus in cacti:
            if isinstance(cactus, list):
                for el in cactus:
                    el['x'] -= cactus_speed
                    if not el['passed'] and el['x'] + el['width'] < dino_x:
                        el['passed'] = True
                        obstacles_passed += 1
            else:
                cactus['x'] -= cactus_speed
                if not cactus['passed'] and cactus['x'] + cactus['width'] < dino_x:
                    cactus['passed'] = True
                    obstacles_passed += 1

        # Удаление вышедших за экран препятствий
        filtered_cacti = []
        for c in cacti:
            if isinstance(c, list): 
                filtered_cacti.extend([item for item in c if item['x'] + item['width'] > 0])
            else:  
                if c['x'] + c['width'] > 0:
                    filtered_cacti.append(c)
        cacti = filtered_cacti

        # Поиск ближайшего препятствия
        nearest = None
        for cactus in cacti:
            if cactus['x'] + cactus['width'] >= dino_x:
                nearest = cactus
                break

        # Обновление динозавров
        for dino in population.dinos:
            if dino.alive:
                dino.update(nearest)
                # Проверка столкновений
                for cactus in cacti:
                    height = duck_height if dino.is_ducking else dino_height
                    dino_rect = pygame.Rect(dino_x, dino.y + (dino_height - height), dino_width, height)
                    cactus_rect = pygame.Rect(cactus['x'], cactus['y'], cactus['width'], cactus['height'])
                    if dino_rect.colliderect(cactus_rect):
                        dino.alive = False

        # Эволюция при гибели всех динозавров
        if population.all_dead():
            population.evolve()
            cacti = [spawn_cactus()]
            cactus_timer = random.randint(60, 120)
            obstacles_passed = 0
            time_alive = 0
            cactus_speed = 5

        # Обновление рекорда
        if obstacles_passed > max_obstacles_passed:
            max_obstacles_passed = obstacles_passed
            
        # Отрисовка кадра
        draw(cacti, obstacles_passed, max_obstacles=max_obstacles_passed)


# Инициализация Pygame и констант
pygame.init()
WIDTH, HEIGHT = 800, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Google Dino AI Evolution")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)

# Параметры игры
POPULATION_SIZE = 500
MUTATION_RATE = 0.1
dino_width, dino_height = 20, 30
dino_x = 50
ground_y = HEIGHT - dino_height - 20
jump_height = 10
gravity = 1
cactus_speed = 5
MAX_SPEED = 20
duck_height = 15

# Вспомогательные объекты
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
obstacles_passed = 0
cacti = [spawn_cactus()]
cactus_timer = random.randint(60, 120)

# Запуск игры
if __name__ == "__main__":
    main()