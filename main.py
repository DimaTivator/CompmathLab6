import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from compmath.differential_equations import *


functions = {
    'f1': f1,
    'f2': f2,
    'f3': f3
}

methods = {
    'Euler': euler,
    'Extended Euler': extended_euler,
    'RK4': runge_kutta_4,
    'Milne': milne,
}

functions_latex = {
    'f1': r'f_1(x, y) = 4x + \frac{y}{3}',
    'f2': r'f_2(x, y) = x^2 + y',
    'f3': r'f_3(x, y) = y \cdot \cos(x)'
}

solutions = {
    'f1': fy1,
    'f2': fy2,
    'f3': fy3
}

consts = {
    'f1': c1,
    'f2': c2,
    'f3': c3
}


def main():
    for func in 'f1', 'f2', 'f3':
        st.latex(f'{functions_latex[func]}')

    equation = st.selectbox('Select equation', ['f1', 'f2', 'f3'])
    func = functions[equation]
    method = st.selectbox('Select method', ['Euler', 'Extended Euler', 'RK4', 'Milne'])

    eps = st.number_input('Eps', step=0.01, value=0.1)
    h = st.number_input('Step', step=0.1, value=0.1)
    left = st.number_input('Left', step=1.0, value=0.0)
    right = st.number_input('Right', step=1.0, value=1.0)
    x0 = st.number_input('x0', step=1.0, value=0.0)
    y0 = st.number_input('y0', step=1.0, value=0.0)

    if left >= right:
        st.warning('Please ensure that the value of "Left" is less than the value of "Right".')

    if st.button('Go'):
        if method in ['Euler', 'Extended Euler', 'RK4']:
            xs, ys = solve_adaptive_step_size(
                DifferentialEquation(x0=x0, y0=y0, a=left, b=right, f=func),
                h,
                epsilon=eps,
                method=methods[method]
            )
        else:
            xs, ys = methods[method](DifferentialEquation(x0=x0, y0=y0, a=left, b=right, f=func), h, eps)

        fig, ax = plt.subplots(figsize=(18, 10))

        ax.scatter(xs, ys, color='black')
        ax.plot(xs, ys, label=method, color='green')

        x_range = np.linspace(min(xs), max(xs), 200)
        ax.plot(x_range, [solutions[equation](x, consts[equation](x0, y0)) for x in x_range], label='Solution', color='orange')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(bottom=min(ys) - 3, top=max(ys) + 3)
        plt.legend()
        st.pyplot(fig)


if __name__ == '__main__':
    main()
