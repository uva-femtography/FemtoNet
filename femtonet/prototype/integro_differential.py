import numpy as np
import scipy.constants as constants

from functools import partial

from .__numerical.__integration import __quad

def __function(x: float, u: float, cache: 'numpy.ndarray'):
    _x = x
    _y = x + 0.01

    return ((1 + np.power(_x / _y, 2)) * cache - 2. * u) / (1 - _x / _y)

def __integral_equation(u: float, x: float, cache: 'numpy.ndarray'):
    f = partial(__function, u=u, cache=cache)

    integral = __quad(func=f, a=x, b=1.)
    return integral

def alpha(x):
    return x

def evolution_equation(u: float, q: float, x: float, cache: 'numpy.ndarray'):
    values = np.zeros_like(x)
    for i, _ in enumerate(x):
        coef = (4. / 3. * alpha(np.exp(q))) / (2. * constants.pi)
        endpoint = u * (2. * np.log1p(1 - x[i]) + 1.5)
        integral = __integral_equation(u, x[i], cache=cache[i])
        values[i] = coef * (integral + endpoint)
    return values
