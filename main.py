import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import sys

def k(u):  # Функция ядра
    if math.fabs(u) <= 1:
        return 0.75*(1-u*u)
    else:
        return 0


def div_half(a, b, f, eps):  # Метод деления отрезка пополам
    while (math.fabs(b-a) > 0.1):
        x1 = (a+b-eps)/2
        x2 = (a+b+eps)/2
        fx1 = f(x1)
        fx2 = f(x2)
        if (fx1 < fx2):
            a = a
            b = x2
        else:
            a = x1
            b = b
    xf = a+b/2
    return xf


def f(c):
    sampleSize = 1001
    x = np.zeros(sampleSize)
    y = np.zeros(sampleSize)
    constanta = c

    for i in range(sampleSize):
        x[i] = i * 0.1

    for i in range(sampleSize):
        y[i] = (math.sin(x[i]) + 0.1 * x[i] + (50 / math.log(x[i] + 10))) * np.random.normal(1, 0.05)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9)

    y_predicted = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        sum1 = 0
        for j in range(x_train.shape[0]):
            sum1 += k((x_test[i] - x_train[j]) / constanta) * y_train[j]
        sum2 = 0
        for j in range(x_train.shape[0]):
            sum2 += k((x_test[i] - x_train[j]) / constanta)
        if sum2 == 0:
            y_predicted[i] = 0
        else:
            y_predicted[i] = sum1 / sum2

    for j in range(sampleSize-1):  # Метод сортировки пузырьком (через кортежи)
        for i in range(x_train.shape[0]-1):
            if x_train[i] > x_train[i+1]:
                (x_train[i], x_train[i+1]) = (x_train[i+1], x_train[i])
                (y_train[i], y_train[i+1]) = (y_train[i+1], y_train[i])
        for i in range(x_test.shape[0] - 1):
            if x_test[i] > x_test[i + 1]:
                (x_test[i], x_test[i + 1]) = (x_test[i + 1], x_test[i])
                (y_predicted[i], y_predicted[i + 1]) = (y_predicted[i + 1], y_predicted[i])
                (y_test[i], y_test[i + 1]) = (y_test[i + 1], y_test[i])

    sum_err = 0
    for i in range(x_test.shape[0]):  # Подсчёт среднеквадратичной ошибки
        sum_err += math.pow(y_test[i] - y_predicted[i], 2)

    return sum_err / x_test.shape[0]  # Среднеквадратичная ошибка (return для f(x) в методе оптимизации)


if __name__ == "__main__":
    np.random.seed(1)
    eps_min = 0.15  # Мин граница
    eps_max = 20  # Макс граница
    x = div_half(eps_min, eps_max, f, 0.03)
    print(x)



