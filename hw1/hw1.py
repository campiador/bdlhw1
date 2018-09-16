import numpy as np


from scipy.special import gamma, kv

from numpy.linalg import inv

## Import plotting tools

import matplotlib.pyplot as plt


def draw_GP_prior_samples_at_x_grid(
        x_grid_G, mean_func, cov_func,
        random_seed=42,
        n_samples=1,l=None, v=None):
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

    # Use consistent random number generator for reproducibility
    prng = np.random.RandomState(int(random_seed))
    samples = prng.multivariate_normal(mean_func(x_grid_G),
                                       cov_func(x_grid_G, x_grid_G, l, v), n_samples)

    return samples


def draw_GP_posterior_samples_at_x_grid(
        x_train_N, y_train_N, x_grid_G, mean_func, cov_func,
        sigma=0.1,
        random_seed=42,
        n_samples=1, l=None, v=None):
    """ Draw sample from GP posterior given training data and mean/cov

    Args
    ----
    x_train_N : 1D array, size n_train_examples (N)
        Each entry i provides the x value observed at training example i
    y_train_N : 1D array, size n_train_examples (N)
        Each entry i provides the y value observed at training example i
    sigma : scalar float
        Specifies the standard deviation of the likelihood.
        y_i drawn from a 1D Normal with mean f(x_i) and std. dev. \sigma.
    Other args same as earlier function: draw_GP_prior_samples_at_x_grid

    Returns
    -------
    f_SG : 2D array, n_samples (S) x n_grid_pts (G)
        Contains sampled function values at each point of x_grid

    """

    prng = np.random.RandomState(int(random_seed))
    # x_star_samples = prng.multivariate_normal(mean_func(x_grid_G),
    #                                    cov_func(x_grid_G, x_grid_G, l, v), n_samples)


    mean_post = np.dot(

                    np.dot(

                                cov_func(x_grid_G, x_train_N, l, v),

                                inv(cov_func(x_train_N, x_train_N, l, v) + np.multiply(sigma, np.identity(len(x_train_N))))
                            )
                    ,
                y_train_N)


    cov_func_post = cov_func(x_grid_G, x_grid_G, l, v) - np.dot(

                np.dot(

                    cov_func(x_grid_G, x_train_N, l, v),

                    inv(cov_func(x_train_N, x_train_N, l, v) + np.multiply(sigma, np.identity(len(x_train_N))))
                )
                ,
                cov_func(x_train_N, x_grid_G, l, v))



    # print(mean_post.shape)
    # print(cov_func_post.shape)

    posterior_samples =  prng.multivariate_normal(mean_post,cov_func_post, n_samples)

    return posterior_samples

def sqexp_kernel_func(x_vector, xp_vector, l, v):

    cov = [
            [np.exp(-(x - xp)**2 / (2 * l**2))
                for xp in xp_vector ]
            for x in x_vector
    ]

    cov = np.asarray(cov)

    # print(cov.shape)

    return cov

def matern_kernel_func(x_vector, xp_vector, l, v):

    cov = [[(2**(1-v) / gamma(v)) * (np.sqrt(2 * v)
                                     * max(abs(x - xp), 1e-15) / l) ** v * kv(v, (np.sqrt(2 * v) *
                                       max(abs(x - xp), 1e-15) / l)) for xp in xp_vector] for x in x_vector]

    cov = np.asarray(cov)

    return cov

def mean_zero(x_vector):
    return [0 for _ in x_vector]

def problem1():

    # number of evenly-space test points
    G = 200
    x_grid_G = np.linspace(-20, 20, G)

    # SQE kernel
    samples_sqexp1 = draw_GP_prior_samples_at_x_grid(x_grid_G, mean_zero, sqexp_kernel_func,
                                    42, 5, 0.25, None)
    s1 = samples_sqexp1.transpose()

    samples_sqexp2 = draw_GP_prior_samples_at_x_grid(x_grid_G, mean_zero, sqexp_kernel_func,
                                                    42, 5, 1, None)
    s2 = samples_sqexp2.transpose()

    samples_sqexp3 = draw_GP_prior_samples_at_x_grid(x_grid_G, mean_zero, sqexp_kernel_func,
                                                    42, 5, 4, None)
    s3 = samples_sqexp3.transpose()


    fig_h, subplot_grid = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=False, squeeze=False)

    subplot_grid[0, 0].plot(x_grid_G, s1)
    subplot_grid[0, 0].set_xlabel('x')
    subplot_grid[0, 0].set_ylabel('samples for l = 0.25')

    subplot_grid[0, 1].plot(x_grid_G, s2)
    subplot_grid[0, 1].set_xlabel('x')
    subplot_grid[0, 1].set_ylabel('samples for l = 1')

    subplot_grid[0, 2].plot(x_grid_G, s3)
    subplot_grid[0, 2].set_xlabel('x')
    subplot_grid[0, 2].set_ylabel('samples for l = 4')


    plt.show()

    #Matern kernel

    l_list = [0.25, 1, 4]
    v_list = [0.50, 2, 8]

    plot_prior_data(l_list, v_list, x_grid_G)


def plot_prior_data(l_list, v_list, x_grid_G):

    fig_h, subplot_grid = plt.subplots(
        nrows=len(v_list), ncols=len(l_list), sharex=True, sharey=False, squeeze=False)


    for index_l, input_l in enumerate(l_list):
        for index_v, input_v in enumerate(v_list):
            samples_matern = draw_GP_prior_samples_at_x_grid(x_grid_G, mean_zero, matern_kernel_func,
                                                             42, 5, input_l, input_v)
            m_t = samples_matern.transpose()

            subplot_grid[index_v, index_l].plot(x_grid_G, m_t)
            subplot_grid[index_v, index_l].set_xlabel('x')
            subplot_grid[index_v, index_l].set_ylabel('l={}, v ={}'.format(input_l, input_v))
    plt.show()


# problem1()

def problem2():
    # number of evenly-space test points
    G = 200
    x_grid_G = np.linspace(-20, 20, G)

    x_train_N = np.asarray([-2., -1.8, -1., 1., 1.8, 2.])
    y_train_N = np.asarray([-3., 0.2224, 3., 3., 0.2224, -3.])



    #note: change as needed
    l_list = [0.25, 1, 4]
    # v_list = [0.50, 2, 8]
    v_list = [1]

    plot_posterior_data(x_train_N, y_train_N, l_list, v_list, x_grid_G)



def plot_posterior_data(x_train_N, y_train_N, l_list, v_list, x_grid_G):

    fig_h, subplot_grid = plt.subplots(
        nrows=len(v_list), ncols=len(l_list), sharex=True, sharey=False, squeeze=False)

    for index_l, input_l in enumerate(l_list):
        for index_v, input_v in enumerate(v_list):
            #note: change the funciton as needed
            samples_matern = draw_GP_posterior_samples_at_x_grid(x_train_N, y_train_N, x_grid_G, mean_zero,
                                                                 sqexp_kernel_func, 0.1, 42, 5, input_l, input_v)
            m_t = samples_matern.transpose()

            subplot_grid[index_v, index_l].plot(x_grid_G, m_t)
            subplot_grid[index_v, index_l].set_xlabel('x')
            subplot_grid[index_v, index_l].set_ylabel('l={}, v ={}'.format(input_l, input_v))
    plt.show()


problem2()
