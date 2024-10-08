"""
Module containing functions for numerical integration using the trapezoidal rule.

And for evaluating specific mathematical functions with Numba JIT compilation.
"""

import math
import time

import numba as nb


def python_quad_trap(f, a, b, N):
    """
    Calculate the integral of a function using the trapezoidal rule.

    Parameters
    ----------
    f (function): The function to integrate.
    a (float): The start point of the interval.
    b (float): The end point of the interval.
    N (int): The number of subintervals.

    Returns
    -------
    float: The approximate integral of the function.
    """
    h = (b - a) / N
    integral = h * (f(a) + f(b)) / 2
    for k in range(N):
        xk = (b - a) * k / N + a
        integral = integral + h * f(xk)
    return integral


@nb.njit
def jitted_quad_trap(f, a, b, N):
    """
    Calculate the integral of a function using the trapezoidal rule.

    Parameters
    ----------
    f (function): The function to integrate.
    a (float): The start point of the interval.
    b (float): The end point of the interval.
    N (int): The number of subintervals.

    Returns
    -------
    float: The approximate integral of the function.
    """
    h = (b - a) / N
    integral = h * (f(a) + f(b)) / 2
    for k in range(N):
        xk = (b - a) * k / N + a
        integral = integral + h * f(xk)
    return integral


def python_integrand(x):
    """
    Compute the value of the expression e^x - 10.

    Parameters
    ----------
    x (float): The exponent to which e is raised.

    Returns
    -------
    float: The result of the expression e^x - 10.
    """
    return math.exp(x) - 10


@nb.njit
def jitted_integrand(x):
    """
    Compute the value of the expression e^x - 10.

    Parameters
    ----------
    x (float): The exponent to which e is raised.

    Returns
    -------
    float: The result of the expression e^x - 10.
    """
    return math.exp(x) - 10


@nb.njit
def parametrized_integrand(x, p):
    return math.exp(p * x) - 10


def slow_internal_function_definition(p):
    # Very slow: Compilation happens every time g(p) is called.
    @nb.njit(nb.float64(nb.float64))
    def integrand(x):
        return math.exp(p * x) - 10

    return jitted_quad_trap(integrand, -1, 1, 10000)


def capturing_function_definition(p: float):
    """
    Create an integrand function for a given parameter.

    Parameters
    ----------
    p (float): The parameter to use in the integrand function.

    Returns
    -------
    function: A Numba JIT-compiled function that computes the integrand.
    """

    @nb.njit
    def integrand(x):
        return math.exp(p * x) - 10

    return integrand


@nb.njit
def parametrized_integrand(x, p):
    return math.exp(p * x) - 10

## Equally fast method:
def default_argument_function_definition(p: float):
    """
    Create an integrand function for a given parameter.

    Parameters
    ----------
    p (float): The parameter to use in the integrand function.

    Returns
    -------
    function: A Numba JIT-compiled function that computes the integrand.
    """
    @nb.njit
    def integrand(x):
        return parametrized_integrand(x, p)

    return integrand


def partial_application_function_definition(func, p: float):
    """
    Create a partially applied function with a fixed parameter `p`.

    Parameters
    ----------
    func : callable
        The function to be partially applied. It should take two arguments.
    p : float
        The parameter to be fixed in the partially applied function.

    Returns
    -------
    callable
        A Numba JIT-compiled function that takes a single argument `x` and
        applies `func` with `x` and the fixed parameter `p`.
    """
    # A **kwargs-based solution does not work with Numba JIT.

    @nb.njit
    def integrand_partial_application(x):
        return func(x, p=p)

    return integrand_partial_application

integrand_capturing_arg = capturing_function_definition(1)

integrand_default_arged = default_argument_function_definition(1)

integrand_partial_application = partial_application_function_definition(
    parametrized_integrand, p=1
)

def python_quadrature_benchmark(quad_func, num_applications):
    start_time = time.time()
    for i in range(num_applications):
        q1 = python_quad_trap(quad_func, -1, 1, 10000)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_applications

    print(f"On average, python quadrature took: {avg_time:.4e} seconds.")


def jit_quadrature_benchmark(quad_func, num_applications):
    # Warm up JIT:
    q1 = jitted_quad_trap(quad_func, -1, 1, 10000)

    start_time = time.time()
    for i in range(num_applications):
        q1 = jitted_quad_trap(quad_func, -1, 1, 10000)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_applications

    print(f"On average, jitted quadrature took: {avg_time:.4e} seconds.")


num_applications = int(1.0e4)

print("Python integrand:")
python_quadrature_benchmark(python_integrand, num_applications)

print("Jitted integrand:")
jit_quadrature_benchmark(jitted_integrand, num_applications)

print("\nCapturing argument:")
jit_quadrature_benchmark(integrand_capturing_arg, num_applications)

print("\nSetting Default argument:")
jit_quadrature_benchmark(integrand_default_arged, num_applications)

print("\nPartial application:")
jit_quadrature_benchmark(integrand_partial_application, num_applications)


