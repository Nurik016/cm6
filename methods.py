def euler_method(f, x0, y0, h, n):
    """
    Solves the differential equation dy/dx = f(x, y) using Euler's method.

    Parameters:
    f  - Function representing dy/dx.
    x0 - Initial x value.
    y0 - Initial y value.
    h  - Step size.
    n  - Number of steps.

    Returns:
    x_vals, y_vals - Lists of x and y values computed using Euler's method.
    """
    x_vals = [x0]
    y_vals = [y0]

    for i in range(n):
        y0 = y0 + h * f(x0, y0)
        x0 = x0 + h
        x_vals.append(x0)
        y_vals.append(y0)

    return x_vals, y_vals


def modified_euler(f, x0, y0, x_end, h):
    """
    Solves the ODE y' = f(x, y) using Modified Euler's Method.

    Parameters:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        h: Step size

    Returns:
        List of (x, y) values.
    """
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < x_end:
        y_predictor = y + h * f(x, y)  # Predictor step
        y_corrector = y + (h / 2) * (f(x, y) + f(x + h, y_predictor))  # Corrector step

        x += h
        y = y_corrector

        x_values.append(x)
        y_values.append(y)

    return list(zip(x_values, y_values))


def runge_kutta_3rd(f, x0, y0, x_end, h):
    """
    Solves the ODE y' = f(x, y) using the Third-Order Runge-Kutta Method.

    Parameters:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        h: Step size

    Returns:
        List of (x, y) values.
    """
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < x_end:
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h, y - k1 + 2 * k2)

        y += (1 / 6) * (k1 + 4 * k2 + k3)
        x += h

        x_values.append(x)
        y_values.append(y)

    return list(zip(x_values, y_values))


def runge_kutta_4th(f, x0, y0, x_end, h):
    """
    Solves the ODE y' = f(x, y) using the Fourth-Order Runge-Kutta Method.

    Parameters:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_end: Final x value
        h: Step size

    Returns:
        List of (x, y) values.
    """
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < x_end:
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        y += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h

        x_values.append(x)
        y_values.append(y)

    return list(zip(x_values, y_values))


def newton_forward_difference_first_order(x_values, y_values):
    """
    Computes the first-order derivative using Newton's Forward Difference Formula.

    Parameters:
        x_values: List of x values (must be equally spaced).
        y_values: List of y values corresponding to f(x).

    Returns:
        List of first-order derivatives at each point.
    """
    h = x_values[1] - x_values[0]  # Assuming equal spacing
    n = len(y_values)

    first_order_derivatives = []

    for i in range(n - 1):  # Since forward difference reduces the number of points
        f_prime = (y_values[i + 1] - y_values[i]) / h
        first_order_derivatives.append(f_prime)

    return first_order_derivatives


def newton_forward_difference_second_order(x_values, y_values):
    """
    Computes the second-order derivative using Newton's Forward Difference Formula.

    Parameters:
        x_values: List of x values (must be equally spaced).
        y_values: List of y values corresponding to f(x).

    Returns:
        List of second-order derivatives at each point.
    """
    h = x_values[1] - x_values[0]  # Assuming equal spacing
    n = len(y_values)

    second_order_derivatives = []

    for i in range(n - 2):  # Second-order differences reduce the number of points further
        f_double_prime = (y_values[i + 2] - 2 * y_values[i + 1] + y_values[i]) / (h ** 2)
        second_order_derivatives.append(f_double_prime)

    return second_order_derivatives