import numpy as np

def example1():
    x = [-1, 0, 1]

    N = len(x)

    m = [1] * N

    k = np.identity(N)


    sample_model = np.random.multivariate_normal(m, k)





example1()

