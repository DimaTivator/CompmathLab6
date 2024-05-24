import numpy as np
from compmath.differential_equations import DifferentialEquation
from typing import Tuple


def euler(equation: DifferentialEquation, h: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the Euler method with adaptive step size.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The initial step size for the Euler method.
        epsilon (float): The desired tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    xs = [x0]
    ys = [y0]
    h = float(h)
    x = x0
    y = y0

    while x < b:
        k1 = h * f(x, y)
        x_next = x + h
        y_next = y + k1
        k2 = h * f(x_next, y_next)
        y_next = y + 0.5 * (k1 + k2)

        # Compute the estimated error using Runge-Kutta method
        k3 = h * f(x + 0.5 * h, y + 0.25 * k1 + 0.25 * k2)
        k4 = h * f(x + h, y - k2 + 2 * k3)
        error = abs((k1 - k2 + k3 - k4) / 3)

        # Update step size based on error
        if error < epsilon:
            xs.append(x_next)
            ys.append(y_next)
            x = x_next
            y = y_next
        h *= min(max(0.84 * (epsilon / error) ** 0.25, 0.1), 4.0)  # Adjust step size

    return np.array(xs), np.array(ys)


def extended_euler(equation: DifferentialEquation, h: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the Extended Euler method with adaptive step size.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The initial step size for the Extended Euler method.
        epsilon (float): The desired tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    xs = [x0]
    ys = [y0]
    h = float(h)
    x = x0
    y = y0

    while x < b:
        k1 = h * f(x, y)
        x_next = x + h
        y_next = y + k1
        k2 = h * f(x_next, y_next)
        y_next = y + 0.5 * (k1 + k2)

        # Compute the estimated error using Runge-Kutta method
        k3 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k4 = h * f(x + h, y + k3)
        error = abs((k1 - k2 + k3 - k4) / 3)

        # Update step size based on error
        if error < epsilon:
            xs.append(x_next)
            ys.append(y_next)
            x = x_next
            y = y_next
        h *= min(max(0.84 * (epsilon / error) ** 0.25, 0.1), 4.0)  # Adjust step size

    return np.array(xs), np.array(ys)


def runge_kutta_4(equation: DifferentialEquation, h: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the fourth-order Runge-Kutta method (RK4) with adaptive step size.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The initial step size for the RK4 method.
        epsilon (float): The desired tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    xs = [x0]
    ys = [y0]
    h = float(h)
    x = x0
    y = y0

    while x < b:
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Compute the estimated error using Runge-Kutta method
        k5 = h * f(x + 0.5 * h, y + 0.25 * k1 + 0.25 * k2)
        k6 = h * f(x + h, y - k2 + 2 * k3)
        error = abs((k1 - k4 + k5 - k6) / 3)

        # Update step size based on error
        if error < epsilon:
            xs.append(x + h)
            ys.append(y_next)
            x += h
            y = y_next
        h *= min(max(0.84 * (epsilon / error) ** 0.25, 0.1), 4.0)  # Adjust step size

    return np.array(xs), np.array(ys)
