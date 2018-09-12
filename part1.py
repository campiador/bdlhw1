import numpy as np


from scipy.special import gamma, kv


def draw_GP_prior_samples_at_x_grid(
        x_grid_G, mean_func, cov_func,
        random_seed=42,
        n_samples=1):
    """ Draw sample from GP prior given mean/cov functions

    Args
    ----
    x_grid_G : 1D array, size n_grid_pts (G)
        Contains G specific x_i values to evaluate function at
    mean_func : function, maps (1D arr size A) to (1D arr size A)
        Computes mean value $m(x_i)$ at each input x_i value
    cov_func : function, maps (1D arr size A, 1D arr size B) to (2D AxB)
        Computes covariance (kernel) value at each pair of inputs.
    random_seed : int
        See for the random number generator
    n_samples : int
        Number of samples to draw from the prior

    Returns
    -------
    f_SG : 2D array, n_samples (S) x n_grid_pts (G)
        Contains sampled function values at each point of x_grid
    """
    # TODO compute mean at each grid point

    # TODO compute covariance matrix across grid points

    # Use consistent random number generator for reproducibility
    prng = np.random.RandomState(int(random_seed))
    samples = prng.multivariate_normal(mean_func(x_grid_G),
                                       cov_func(x_grid_G, x_grid_G), n_samples)

    return samples

def problem1():


    # x = [-1, 0, 1]
    #
    # N = len(x)
    #
    # m = [1] * N
    #
    # k = np.identity(N)
    #
    # sample_model = np.random.multivariate_normal(m, k)
    #
    # # Use consistent random number generator for reproducibility
    # prng = np.random.RandomState(int(1))
    #
    # x = prng.multivariate_normal(m, k)
    #
    # print(x.shape)
    # print(x)

    def sqexp_kernel_func(x_vector, xp_vector, l=0.25):

        cov = [
                [np.exp(-(x - xp)**2 / (2 * l**2))
                    for xp in xp_vector ]
                for x in x_vector
        ]

        cov = np.asarray(cov)

        print(cov.shape)

        return cov

    def matern_kernel_func(x_vector, xp_vector):
        l = 0.25
        v = 0.5



        cov = [[(2**(1-v) / gamma(v)) * (np.sqrt(2 * v)
                                         * max(abs(x - xp), 1e-15) / l) ** v * kv(v, (np.sqrt(2 * v) *
                                           max(abs(x - xp), 1e-15) / l)) for xp in xp_vector] for x in x_vector]

        cov = np.asarray(cov)

        return cov

    def mean_zero(x_vector):
        return [0 for _ in x_vector]


    # number of evenly-space test points
    G = 200 #TODO: should be 200
    x_grid_G = np.linspace(-20, 20, G)



    # samples_sqexp = draw_GP_prior_samples_at_x_grid(x_grid_G, mean_zero, sqexp_kernel_func,
    #                                 42, 5)

    samples_matern = draw_GP_prior_samples_at_x_grid(x_grid_G, mean_zero, matern_kernel_func,
                                    42, 5)

    # pass


    #
    #

problem1()



