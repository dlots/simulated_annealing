from collections import defaultdict
import random
import numpy.random as nprandom
from math import exp
from copy import deepcopy

import glob
import numpy as np
from itertools import combinations
from random import randint
from pathlib import Path
import copy

DEBUG = True


def similarity_mas(data):
    m, p = data.shape
    similarity = []

    for i in range(0, p - 1):
        for j in range(i + 1, p):
            a_ij = 0
            b_ij = 0
            c_ij = 0

            for num in range(m):
                p1 = data[num][i]
                p2 = data[num][j]

                if p1 == 1 and p2 == 1:
                    a_ij += 1

                elif p1 == 1 and p2 == 0:
                    b_ij += p1

                elif p1 == 0 and p2 == 1:
                    c_ij += p2

            S = a_ij / (a_ij + b_ij + c_ij)

            if S != 0:
                similarity.append(([i, j], S))

    similarity.sort(key=lambda i: i[1], reverse=True)
    return (similarity)


def create_interval(t):
    n = 100
    ret = []
    for i in range(t-1):
            ret.append(((int)((i+1)*(n-1)/t))/100)
    ret.append(1)
    return ret


def divide_parts(data, similarity, n):  # n - количество интервалов
    m, p = data.shape

    interval = create_interval(n)
    parts_in_work = defaultdict(list)
    list_sim = similarity

    for i in list_sim:
        for j in range(n):  # пробегаем по всем интервалам
            if i[1] <= interval[j]:
                if i[0][0] not in parts_in_work:
                    parts_in_work[(i[0][0])] = j

                if i[0][1] not in parts_in_work:
                    parts_in_work[(i[0][1])] = j

    div_parts = []
    for i in range(n):
        div_parts.append([])

    for part in parts_in_work:
        div_parts[parts_in_work[part]].append(part)  # заносим результат в кластеры

    if len(parts_in_work) != p:  # если есть детали, не вошедшие ни в 1 кластер
        missing = list(set(i for i in range(p)) - set(key for key in parts_in_work))
        for mis in missing:
            a = random.randint(0, n - 1)  # рандомно помещаем их в кластеры
            div_parts[a].append(mis)

    div_parts = [value for value in div_parts if len(value) != 0]  # убираем все пустые кластеры

    return (div_parts)


def get_indices(lst, elem):
    return [i for i, x in enumerate(lst) if x == elem]


def divide_machines(data, div_parts):
    sum_units = list(map(sum, data))
    m, p = data.shape
    mas = []

    for i in range(m):
        temp = []
        for j in div_parts:  # перебираем кластеры
            v = 0
            e = 0
            for part in j:
                if data[i][part] == 0:
                    v += 1
                else:
                    e += 1

            e = sum_units[i] - e  # единицы, которые не вошли в кластер
            temp.append(v + e)
        mas.append(temp)

    n = len(div_parts)
    div_machines = []
    for i in range(n):
        div_machines.append([])

    for i in range(len(mas)):
        minimum = min(mas[i])  # ищем минимальную погрешность
        positions = get_indices(mas[i], minimum)  # на случай если минимальных погрешностей несколько
        placed = False
        for pos in positions:
            if len(div_machines[pos]) == 0:
                div_machines[pos].append(i)
                placed = True
        if not placed:
            position = random.choices(positions, k=1)[0]
            div_machines[position].append(i)

    #div_machines = [value for value in div_machines if len(value) != 0]  # убираем все пустые кластеры
    return (div_machines)


def initial_solution(data, n): #n - количество интервалов, на которое постараемся разделить
    similarity = similarity_mas(data)
    div_parts = divide_parts(data, similarity, n)
    div_machines = divide_machines(data, div_parts)
    return(div_parts, div_machines)


def core_function(div_parts, div_machines, data):
    n_1 = 0  # количество единиц в кластере
    n_1_out = 0  # количество единиц, которые никуда не попали
    n_0_in = 0  # количество нулей в кластере

    sum_units = list(map(sum, data))

    for workshop in range(len(div_machines)):
        for i in div_machines[workshop]:
            sum_1 = 0  # сумма единиц в кластере в 1 строке

            for j in div_parts[workshop]:
                if data[i][j] == 1:
                    sum_1 += 1
                else:
                    n_0_in += 1

            n_1 += sum_1
            n_1_out += sum_units[i] - sum_1

    f = (n_1 - n_1_out) / (n_1 + n_0_in)
    # print(n_1, n_1_out, n_0_in)
    return (f)


class SimulatedAnnealing:
    def __init__(self, data, initial_temperature, final_temperature, cooling_rate, max_iterations, exchange_period,
                 max_stagnant):
        self.data = data
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.max_stagnant = max_stagnant
        self.exchange_period = exchange_period
        self.current_cells = 2
        self.optimal_cells = 2
        self.counter = 0
        self.counter_mc = 0
        self.counter_trapped = 0
        self.counter_stagnant = 0
        self.current_solution = None
        self.best_solution_current_cells = None
        self.best_solution = None

    def single_move(self, solution):
        parts, machines = deepcopy(solution)
        length = 1
        while (length < 2):
            cell_index = nprandom.randint(0, len(parts))
            cell = parts[cell_index]
            length = len(cell)
        part = cell[nprandom.randint(0, len(cell))]
        cell.remove(part)
        max_function_value = -9999999
        best_move = None
        for i in range(len(parts)):
            if i == cell_index:
                continue
            parts[i].append(part)
            new_machines = divide_machines(self.data, parts)
            new_function_value = core_function(parts, new_machines, self.data)
            if new_function_value > max_function_value:
                max_function_value = new_function_value
                best_move = deepcopy(parts), new_machines
            parts[i].pop()
        return best_move

    def exchange_move(self, solution):
        parts, machines = deepcopy(solution)
        length = 1
        while (length < 2):
            cell_index = nprandom.randint(0, len(parts))
            cell = parts[cell_index]
            length = len(cell)
        part = cell[nprandom.randint(0, len(cell))]
        cell.remove(part)
        max_function_value = -9999999
        best_move = None
        for i in range(len(parts)):
            if i == cell_index:
                continue
            replacement_part = parts[i][nprandom.randint(0, len(parts[i]))]
            parts[i].remove(replacement_part)
            cell.append(replacement_part)
            parts[i].append(part)
            new_machines = divide_machines(self.data, parts)
            new_function_value = core_function(parts, new_machines, self.data)
            if new_function_value > max_function_value:
                max_function_value = new_function_value
                best_move = deepcopy(parts), new_machines
            parts[i].pop()
            parts[i].append(replacement_part)
            cell.remove(replacement_part)
        return best_move

    def temperature_iteration(self):
        #print('Temperature: ', self.temperature)
        while self.counter_mc < self.max_iterations and self.counter_trapped < (self.max_iterations / 2):
            neighborhood_solution = self.single_move(self.current_solution)
            if self.counter % self.exchange_period == 0:
                neighborhood_solution = self.exchange_move(neighborhood_solution)
            neighborhood_solution_function_value = core_function(*neighborhood_solution, self.data)
            best_solution_current_cells_function_value = core_function(*self.best_solution_current_cells, self.data)
            if neighborhood_solution_function_value > best_solution_current_cells_function_value:
                self.best_solution_current_cells = neighborhood_solution
                self.current_solution = neighborhood_solution
                self.counter_stagnant = 0
            elif neighborhood_solution_function_value == best_solution_current_cells_function_value:
                self.current_solution = neighborhood_solution
                self.counter_stagnant += 1
            else:
                delta = neighborhood_solution_function_value - core_function(*self.current_solution, self.data)
                if exp(delta / self.temperature) > nprandom.uniform():
                    self.current_solution = neighborhood_solution
                    self.counter_trapped = 0
                else:
                    self.counter_trapped += 1
            self.counter_mc += 1

    def search_with_current_cells(self):
        #print('Cells: ', self.current_cells)
        self.counter = 0
        self.counter_mc = 0
        self.counter_trapped = 0
        self.counter_stagnant = 0
        self.temperature = self.initial_temperature
        if self.current_cells > 2:
            self.current_solution = initial_solution(self.data, self.current_cells)
            self.best_solution_current_cells = self.current_solution
        self.temperature_iteration()
        while not (self.temperature <= self.final_temperature or self.counter_stagnant > self.max_stagnant):
            self.temperature *= self.cooling_rate
            self.counter_mc = 0
            self.counter += 1
            self.temperature_iteration()

    def compute(self):
        if DEBUG:
            nprandom.seed(42)
        else:
            nprandom.seed()
        self.current_cells = 2
        self.current_solution = initial_solution(self.data, self.current_cells)
        self.best_solution_current_cells = self.current_solution
        self.best_solution = self.best_solution_current_cells
        self.optimal_cells = self.current_cells
        self.search_with_current_cells()
        while core_function(*self.best_solution_current_cells, self.data) > core_function(*self.best_solution, self.data):
            self.best_solution = self.best_solution_current_cells
            self.optimal_cells = self.current_cells
            self.current_cells += 1
            self.search_with_current_cells()
        return self.best_solution


def qap_reader(path):
    with open(path, "r") as f:
        m, p = f.readline().strip().split()
        m = int(m)
        p = int(p)
        matrix = np.zeros((m, p))

        for i in range(m):
            mas = list(map(int, f.readline().split()))
            for j in range(1, len(mas)):
                matrix[mas[0] - 1][mas[j] - 1] = 1
        return (matrix)

if __name__ == '__main__':
    benchmarks = glob.glob('cfp_data/*')
    data = list()  # все данные
    for file in benchmarks:
        data.append(qap_reader(file))

    # parameters
    initial_temperature = 500
    final_temperature = 200
    cooling_rate = 0.8
    max_iterations = 3000
    exchange_period = 10
    max_stagnant = 50

    #divide_parts(data[0])
    #parts = [[17, 1, 7, 5, 12, 3, 0, 19, 13, 10, 15, 2, 18, 8, 14, 11, 9, 6, 16], [4]]
    #print(divide_machines(data[0], parts))
    algorithm = SimulatedAnnealing(data[0], initial_temperature, final_temperature, cooling_rate, max_iterations, exchange_period, max_stagnant)
    algorithm.compute()
    #sol = initial_solution(data[0], 2)
    #print(sol)
    #print(algorithm.exchange_move(sol))
