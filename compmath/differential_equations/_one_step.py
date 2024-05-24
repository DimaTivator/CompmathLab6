import numpy as np
from compmath.differential_equations import DifferentialEquation
from typing import Tuple


def euler(equation: DifferentialEquation, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the Euler method.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The step size for the Euler method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    n_steps = int((b - a) / h)
    xs = np.linspace(a, b, n_steps + 1, dtype=np.float64)
    ys = np.zeros(n_steps + 1, dtype=np.float64)
    ys[0] = y0

    for i in range(1, n_steps + 1):
        ys[i] = ys[i - 1] + h * f(xs[i - 1], ys[i - 1])

    return xs, ys


def extended_euler(equation: DifferentialEquation, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the Extended Euler method (also known as Improved Euler or Heun's method).

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The step size for the Extended Euler method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    n_steps = int((b - a) / h)
    xs = np.linspace(a, b, n_steps + 1, dtype=np.float64)
    ys = np.zeros(n_steps + 1, dtype=np.float64)
    ys[0] = y0

    for i in range(1, n_steps + 1):
        k1 = h * f(xs[i - 1], ys[i - 1])
        k2 = h * f(xs[i], ys[i - 1] + k1)
        ys[i] = ys[i - 1] + 0.5 * (k1 + k2)

    return xs, ys





