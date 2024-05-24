import numpy as np
from compmath.differential_equations import DifferentialEquation
from typing import Tuple


def milne(equation: DifferentialEquation, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using Milne's method.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The step size for Milne's method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    n_steps = int((b - a) / h)
    xs = np.linspace(a, b, n_steps + 1, dtype=np.float64)
    ys = np.zeros(n_steps + 1, dtype=np.float64)
    ys[0] = y0

    # Use RK4 to generate initial values for Milne's method
    for i in range(1, 4):
        k1 = h * f(xs[i - 1], ys[i - 1])
        k2 = h * f(xs[i - 1] + 0.5 * h, ys[i - 1] + 0.5 * k1)
        k3 = h * f(xs[i - 1] + 0.5 * h, ys[i - 1] + 0.5 * k2)
        k4 = h * f(xs[i - 1] + h, ys[i - 1] + k3)
        ys[i] = ys[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    for i in range(3, n_steps):
        ys[i + 1] = ys[i - 3] + 4 * h / 3 * (2 * f(xs[i], ys[i]) - f(xs[i - 1], ys[i - 1]) + 2 * f(xs[i - 2], ys[i - 2]))

    return xs, ys
