# Лабораторная работа №3: Алгоритм имитации обжига
### 18ПМИ Богородицкая Екатерина, Сазанов Дмитрий, Селивановская Дарья


```python
import glob
import numpy as np
from tqdm import tqdm
from itertools import combinations
from random import randint
from pathlib import Path
from collections import defaultdict
import copy
import random
import time
```


```python
def QAPreader(path):
    with open(path, "r") as f:
        m, p = f.readline().strip().split()
        m = int(m)
        p = int(p)
        matrix = np.zeros((m, p))
        
        for i in range(m):
            mas = list(map(int, f.readline().split()))
            for j in range(1, len(mas)):
                matrix[mas[0]-1][mas[j]-1] = 1
        return(matrix)        
```


```python
benchmarks = glob.glob('cfp_data/*')

data = list() #все данные
for file in benchmarks:
    data.append(QAPreader(file))
```


```python
for i in data[0]:
    print(i)
```

    [1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0.]
    [0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
    [0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0. 1. 1. 0. 1. 0. 1. 0. 0.]
    [0. 0. 0. 0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    [1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 1. 1. 1.]
    [0. 0. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 1. 0. 1.]
    [0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 1. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1. 0.]
    [0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1. 1. 0.]
    [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 1. 0. 0.]
    [0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 0.]
    [0. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 0.]
    [0. 0. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0.]
    [1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    [1. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
    


```python
from simulated_annealing import SimulatedAnnealing, core_function
# class SimulatedAnnealing parameters: data, initial_temperature, final_temperature, cooling_rate, max_iterations,
#                                      exchange_period, max_stagnant
```


```python
# parameters
initial_temperature = 500
final_temperature = 200
cooling_rate = 0.8
max_iterations = 500
exchange_period = 10
max_stagnant = 10
```


```python
counter = 1
for data_item in data:
    algorithm = SimulatedAnnealing(data_item, initial_temperature, final_temperature, cooling_rate, max_iterations,
                                   exchange_period, max_stagnant)
    start = time.time()
    solution = algorithm.compute()
    elapsed = time.time() - start
    print('benchmark #', counter)
    print('elapsed: ', elapsed)
    print('parts: ', solution[0])
    print('machines: ', solution[1])
    print('core function value: ', core_function(*solution, data_item))
    print('')
    counter+=1
```

    benchmark # 1
    elapsed:  22.52915120124817
    parts:  [[15, 6, 8, 10, 12, 19, 3, 18, 16, 4, 13, 17, 5], [9, 11, 7, 1, 0, 2, 14]]
    machines:  [[0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16], [1, 3, 4, 9, 14, 17, 18, 19]]
    core function value:  0.24056603773584906
    
    benchmark # 2
    elapsed:  45.3608295917511
    parts:  [[2, 3, 33, 23, 34, 4, 26, 30, 0, 11, 29, 6, 22, 28, 8, 18, 19, 17, 9], [24, 7, 14, 1, 16, 5, 39, 27, 36, 32, 20, 10, 12, 37, 13, 25, 31, 15, 38, 35, 21]]
    machines:  [[0, 2, 4, 5, 7, 9, 10, 11, 12, 14, 16, 17, 19, 20, 21], [1, 3, 6, 8, 13, 15, 18, 22, 23]]
    core function value:  0.12236286919831224
    
    benchmark # 3
    elapsed:  109.44049859046936
    parts:  [[7, 48, 35, 44, 43, 12, 34, 39, 47, 38, 31, 14, 15, 30, 41, 3, 9, 13, 28], [40, 17, 20, 21, 4, 29, 33, 32, 6, 26, 42, 25, 27, 0, 46, 36], [24, 49, 37, 19, 45, 8, 10, 23, 2, 22, 5, 18, 1, 11, 16]]
    machines:  [[2, 7, 13, 15, 20, 21, 24, 25, 27, 29], [0, 3, 6, 8, 16, 17, 18, 19, 23, 26], [1, 4, 5, 9, 10, 11, 12, 14, 22, 28]]
    core function value:  0.106
    
    benchmark # 4
    elapsed:  113.37925910949707
    parts:  [[62, 41, 64, 43, 61, 3, 16, 89, 30, 15, 59, 56, 2, 8, 33, 10, 0, 52, 57, 65, 73, 78, 63, 88, 58, 51, 31, 48, 53, 36, 69, 22, 40, 13, 60, 39, 77, 80, 24, 66, 70, 46, 71, 54, 74, 49, 4], [42, 55, 19, 35, 81, 47, 21, 11, 38, 83, 67, 84, 50, 37, 34, 87, 25, 45, 20, 79, 1, 17, 26, 76, 75, 85, 5, 29, 44, 9, 7, 12, 14, 86, 28, 82, 32, 27, 18, 72, 6, 68, 23]]
    machines:  [[0, 2, 13, 25, 26, 27, 28], [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29]]
    core function value:  0.0622154779969651



```python

```
