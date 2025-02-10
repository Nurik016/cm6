import numpy as np

def forward_difference_table(x, y):
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[0, :] = y
    for k in range(1, n):
        for i in range(n - k):
            diff_table[k, i] = diff_table[k - 1, i + 1] - diff_table[k - 1, i]
    return diff_table


def newton_forward_diff_derivative(x, diff_table, x_point, order=1):
    h = x[1] - x[0]
    n = len(x)
    r = (x_point - x[0]) / h
    if order == 1:
        val = 0.0
        factorial = 1
        for k in range(1, n):
            factorial *= k
            c = derivative_coefficient(k, r)
            val += (c / factorial) * diff_table[k, 0]
        return val / h
    elif order == 2:
        val = 0.0
        factorial = 1
        for k in range(2, n):
            factorial *= k
            c = second_derivative_coefficient(k, r)
            val += (c / factorial) * diff_table[k, 0]
        return val / (h ** 2)


def derivative_coefficient(k, r):
    if k == 1:
        return 1.0
    elif k == 2:
        return 2.0 * r - 1.0
    elif k == 3:
        return 3.0 * r ** 2 - 6.0 * r + 2.0
    elif k == 4:
        return 4.0 * r ** 3 - 18.0 * r ** 2 + 24.0 * r - 6.0
    return 0.0


def second_derivative_coefficient(k, r):
    if k == 2:
        return 2.0
    elif k == 3:
        return 6.0 * r - 6.0
    elif k == 4:
        return 12.0 * r ** 2 - 36.0 * r + 24.0
    return 0.0


def f_example(t, y):
    return t + y


t0 = 0.0
y0 = 1.0
h = 0.1
n = 10

x_data = np.array([0, 1, 2, 3, 4], dtype=float)
y_data = x_data ** 2
diff_table = forward_difference_table(x_data, y_data)
x_point = 1.5
dfdx_approx_1 = newton_forward_diff_derivative(x_data, diff_table, x_point, order=1)
dfdx_true_1 = 2 * x_point
dfdx_approx_2 = newton_forward_diff_derivative(x_data, diff_table, x_point, order=2)
dfdx_true_2 = 2.0

print("\n=== Newton's Forward Difference ===")
print(f"x = {x_point}, 1st deriv approx = {dfdx_approx_1:.6f}, actual = {dfdx_true_1:.6f}")
print(f"x = {x_point}, 2nd deriv approx = {dfdx_approx_2:.6f}, actual = {dfdx_true_2:.6f}")
