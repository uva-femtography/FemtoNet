import numpy as np

class FemtoEvolve():

    def __init__(self, function=None, initial_value=None, grid=None, window=None):
        self._default_window_size = 7
        
        self._function = function 
        self._initial_value = initial_value
        self._grid = grid
        self._window = window

    def _shift(self, array, value):
        assert array.shape[0] > 1, "Shift bigger than array length."

        return np.append(array[1:], value)

    def _recurse(self, x, y, length, index=0, h=0.001):
        y[index] -= self._function(x[index], y[index])*h
        index+=1

        if(index <= length):
            self._recurse(x, y, length, index)
  
        return y

    def solver(self):
  
        h = self._grid[1] - self._grid[0]
        y= self._initial_value

        c1 = 1/(2*h)
        c2 = 5/(12*h)
        c3 = 3/(8*h)

        values = np.array([])

        if self._window is not None and type(self._window) == list or type(self._window) == np.ndarray:
            X = np.zeros(self._default_window_size)
            Y = np.array(self._window) if type(self._window) == np.ndarray else np.array(self._window)
        
        else:
            X = np.zeros(self._default_window_size)
            Y = np.zeros(self._default_window_size)

            Y[-1] = self._initial_value

        for i in self._grid:
            X = self._shift(X, i)
            Y = self._recurse(X, Y, length=X.shape[0]-1, index=1, h=h)

            values = np.append(values, y)

            term1 = c1*( Y[6] - 2.*Y[5] + Y[4] )
            term2 = c2*( Y[5] - 3*Y[4] + 3*Y[3] - Y[2] )
            term3 = c3*( Y[4] - 4*Y[3] + 6*Y[2] - 4*Y[1] + Y[0] )
    
            y += h*( self._function(i, y) + term1)
        
            Y = self._shift(Y, y)

        return values