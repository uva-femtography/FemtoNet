import numpy as np

def __simpsons(func=None, a=0, b=0, **kwargs):
    if kwargs is not None:
        h = 0.5*(b-a)
        return (1/3.)*h*(func(a, **kwargs) + 4*func(0.5*(a+b), **kwargs) + func(b, **kwargs))
    else:
        h = 0.5 * (b - a)
        return (1 / 3.) * h * (func(a) + 4 * func(0.5 * (a + b)) + func(b))

def __quad(func=None, a=0, b=0, n_grid_points=100, type='simpsons', **kwargs):
    grid = np.linspace(a, b, n_grid_points)
    value = 0.
    if type is 'simpsons':
        for i in range(grid.shape[0] - 1):
            value += __simpsons(func, grid[i], grid[i+1], **kwargs)
        return value
    else:
        return 0