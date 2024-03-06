# Just for reference as frame or scaffold 
# From https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21


import numpy as np
from typing import Callable

# 1. def gradient decent plustracking
def gradient_descent(start: float, gradient: Callable[[float], float],
                     learn_rate: float, max_iter: int, tol: float = 0.01):
    x = start
    steps = [start]  # history tracking

    for _ in range(max_iter):
        diff = learn_rate*gradient(x)
        if np.abs(diff) < tol:
            break
        x = x - diff
        steps.append(x)  # history tracing
  
    return steps, x



# 2. sample function and it's gradient function
def func1(x:float):
    return x**2-4*x+1

def gradient_func1(x:float):
    return 2*x - 4



# 3. sample function-call 
history, result = gradient_descent(9, gradient_func1, 0.1, 100)

# 4. plot, outputs, some more samples on webpage