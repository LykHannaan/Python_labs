import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Генерация случайных координат и дорог
def generate_random_points(num_points, space_size):
    return np.random.rand(num_points, 3) * space_size

def generate_edges(points, bidirectional=True):
    num_points = len(points)
    edges = []
    # Создание матрицы смежности 
    adjacency_matrix = np.random.rand(num_points, num_points) < 0.5  # 50% вероятность создания дороги
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if adjacency_matrix[i, j]:
                edges.append((i, j))  
                if bidirectional:
                    edges.append((j, i))
    return edges

# Решение задачи коммивояжера с использованием муравьиного алгоритма
class AntColony:
    def __init__(self, points, num_ants, num_iterations, alpha=1, beta=1, evaporation_rate=0.5):
        self.points = points  
        self.num_ants = num_ants  
        self.num_iterations = num_iterations  
        self.alpha = alpha  
        self.beta = beta  
        self.evaporation_rate = evaporation_rate  
        self.pheromone_matrix = np.ones((len(points), len(points)))  
        # Вычисление матрицы расстояний между всеми точками
        self.distance_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)  
        self.best_route = None  
        self.best_distance = float('inf')  

    def total_distance(self, route):
        return self.distance_matrix[route[:-1], route[1:]].sum()

    def update_pheromones(self, all_routes):
        self.pheromone_matrix *= (1 - self.evaporation_rate)  
        for route in all_routes:
            distance = self.total_distance(route)  
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i], route[i + 1]] += 1 / distance

    def run(self):
        for _ in range(self.num_iterations):
            all_routes = []  # Список всех маршрутов
            for _ in range(self.num_ants):
                route = self.construct_route()  # Построение маршрута для муравья
                distance = self.total_distance(route)
                all_routes.append(route) 
                # Проверка, является ли текущий маршрут лучшим
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route  
            self.update_pheromones(all_routes)  

    def construct_route(self):
        # маршрут для одного муравья
        route = [0]  
        visited = set(route) 
        for _ in range(1, len(self.points)): 
            current = route[-1]  
            probabilities = [] 
            for next_point in range(len(self.points)):
                if next_point not in visited: 
                    pheromone = self.pheromone_matrix[current][next_point] ** self.alpha  # Влияние феромонов
                    distance = (1 / self.distance_matrix[current, next_point]) ** self.beta  # Влияние расстояния
                    probabilities.append(pheromone * distance)  # вероятность
                else:
                    probabilities.append(0)  # Если точка уже посещена, вероятность 0
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()  
          
            next_point = np.random.choice(range(len(self.points)), p=probabilities)
            route.append(next_point)  
            visited.add(next_point)  
        return route + [0]  # Вернуться в начальную точку

# Визуализация
def plot_route(points, route):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue')

    # Построение маршрута
    ax.plot(points[route, 0], points[route, 1], points[route, 2], color='red')

    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    plt.title('Best Route Visualization') 
    plt.show()  

# Проверка решения на наборах из 200, 500, 1000 точек
for num_points in [200]:  #500, 1000
    points = generate_random_points(num_points, space_size=100)  # Генерация случайных точек
    
    print("Coordinates of points:")
    for i, point in enumerate(points):
        print(f"Point {i}: (X: {point[0]:.2f}, Y: {point[1]:.2f}, Z: {point[2]:.2f})")

    # Инициализация и запуск алгоритма 
    ant_colony = AntColony(points, num_ants=10, num_iterations=100)
    ant_colony.run()  

    print(f"Best distance for {num_points} points:", ant_colony.best_distance)
    plot_route(points, ant_colony.best_route)  # Визуализация лучшего маршрута


