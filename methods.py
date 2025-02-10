def newton_forward_difference_first_order(x_values, y_values):
    """
    Computes the first-order derivative using Newton's Forward Difference Formula.

    Parameters:
        x_values: List of x values (must be equally spaced).
        y_values: List of y values corresponding to f(x).

    Returns:
        List of first-order derivatives at each point.
    """
    h = x_values[1] - x_values[0]
    n = len(y_values)

    delta = [[0] * n for _ in range(n)]

    for i in range(n):
        delta[i][0] = y_values[i]

    for j in range(1, n):
        for i in range(n - j):
            delta[i][j] = delta[i + 1][j - 1] - delta[i][j - 1]


    first_order_derivatives = (delta[0][1] - (1 / 2) * delta[0][2] + (1 / 3) * delta[0][3]
                  - (1 / 4) * delta[0][4] + (1 / 5) * delta[0][5] - (1 / 6) * delta[0][6]) / h

    return first_order_derivatives


def newton_forward_difference_second_order(x_values, y_values):
    """
    Computes the second-order derivative using Newton's Forward Difference Formula.

    Parameters:
        x_values: List of x values (must be equally spaced).
        y_values: List of y values corresponding to f(x).

    Returns:
        Second-order derivative at x_0.
    """
    h = x_values[1] - x_values[0]
    n = len(y_values)

    delta = [[0] * n for _ in range(n)]

    for i in range(n):
        delta[i][0] = y_values[i]

    for j in range(1, n):
        for i in range(n - j):
            delta[i][j] = delta[i + 1][j - 1] - delta[i][j - 1]

    second_order_derivatives = (delta[0][2] - delta[0][3] + (11 / 12) * delta[0][4]
                         - (5 / 6) * delta[0][5] + (137 / 180) * delta[0][6]) / (h ** 2)

    return second_order_derivatives

