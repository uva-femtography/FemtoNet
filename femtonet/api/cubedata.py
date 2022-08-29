import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass(init=False)
class CubeData:
    data=None
    columns=[]

    def to_json()->'json':
        return self.data.to_json()

    def to_numpy()->'numpy.array()':
        return self.data.to_numpy()

    
    
