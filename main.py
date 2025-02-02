from tabulate import tabulate
from colorama import Fore, Style

from methods import (
    euler_method,
    modified_euler,
    runge_kutta_3rd,
    runge_kutta_4th,
    newton_forward_difference_first_order,
    newton_forward_difference_second_order,
)

# Пример уравнения: dy/dx = x + y
def func(x, y):
    return x**2 + y  # Функция, которая складывает квадрат x и y

# Начальные условия
x0 = 0
y0 = 1
h = 0.1
n = 10
x_end = 1.0

# Тестируем методы решения ОДУ
x_euler, y_euler = euler_method(func, x0, y0, h, n)
xy_mod_euler = modified_euler(func, x0, y0, x_end, h)
xy_rk3 = runge_kutta_3rd(func, x0, y0, x_end, h)
xy_rk4 = runge_kutta_4th(func, x0, y0, x_end, h)

# Вычисляем производные методом конечных разностей
x_vals = [x for x, _ in xy_rk4]
y_vals = [y for _, y in xy_rk4]

first_order_derivatives = newton_forward_difference_first_order(x_vals, y_vals)
second_order_derivatives = newton_forward_difference_second_order(x_vals, y_vals)

# Подготовка данных для таблицы
table = []

# Цвета для методов
method_colors = {
    "Euler's Method": Fore.GREEN,
    "Modified Euler's Method": Fore.BLUE,
    "Runge-Kutta 3rd Order": Fore.YELLOW,
    "Runge-Kutta 4th Order": Fore.CYAN,
    "First-Order Derivatives": Fore.MAGENTA,
    "Second-Order Derivatives": Fore.RED
}

# Добавление результатов для каждого метода с цветами
def format_method_results(method_name, results):
    for x, y in results:
        table.append([method_colors.get(method_name, '' ) + method_name + Style.RESET_ALL,
                      round(x, 2), round(y, 4)])

format_method_results("Euler's Method", zip(x_euler, y_euler))
format_method_results("Modified Euler's Method", xy_mod_euler)
format_method_results("Runge-Kutta 3rd Order", xy_rk3)
format_method_results("Runge-Kutta 4th Order", xy_rk4)

# Добавление производных с цветами
def format_derivative_results(derivatives, label):
    for i, val in enumerate(derivatives):
        table.append([method_colors.get(label, '') + label + Style.RESET_ALL,
                      round(x_vals[i], 2), round(val, 4)])

format_derivative_results(first_order_derivatives, "First-Order Derivatives")
format_derivative_results(second_order_derivatives, "Second-Order Derivatives")

# Печать таблицы с цветами
headers = [Fore.BLACK + "Method" + Style.RESET_ALL, "X", "Y"]
print(tabulate(table, headers=headers, tablefmt="grid"))
