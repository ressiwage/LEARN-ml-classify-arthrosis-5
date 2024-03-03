from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean 


def get_files(p):
    return [f for f in listdir(p) if isfile(join(p, f))]

amounts={}

for t_t in ['test', 'train']:
    for i in list(map(str, range(5))):
        amounts[t_t+i] = len(get_files(''+t_t+'/'+i+'/'))


# Создаем данные для примера
x = [i for i in amounts]   # Порядковые значения на оси x
y = [amounts[i] for i in x] + [mean(amounts[i] for i in x if 'test' in i)] + [mean(amounts[i] for i in x if 'train' in i)]  # Случайные значения для столбцов
x = x +['mtest', 'mtrain']

# Разделяем столбцы на две группы (каждый второй будет красным)
colors = ['green' if 'test' in i else 'blue' for i in x]  # Создаем список цветов для столбцов

# Построение гистограммы
plt.bar(x, y, color=colors)

# Добавляем подписи к столбцам
for i, j in zip(x, y):
    plt.text(i, j, str(j), ha='center', va='bottom')

# Настройка осей и заголовка
plt.xlabel('X')
plt.ylabel('Counts')
plt.title('Гистограмма')
plt.xticks(rotation=45)


# Отображение графика
plt.savefig('anal.png')