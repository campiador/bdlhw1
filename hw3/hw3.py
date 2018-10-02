# BDL HW3 by Behnam Heydarshahi
# September-October 2018

from __future__ import absolute_import

import autograd.numpy as np

np.random.seed(42)

import matplotlib.pyplot as plt

from models.bnn import Bnn
from models.unit import Unit

import autograd


def problem1(part):
    x_train_N = np.asarray([-5.0, -2.50, 0.00, 2.50, 5.0])
    y_train_N = np.asarray([-4.91, -2.48, 0.05, 2.61, 5.09])

    n_samples_list = [1, 10, 100, 1000]

    # mean of w
    m_tilda_list = np.linspace(-3.0, 5.0, 20)


    # log stddev of w
    s_tilda = np.log(0.1)

    # mean of b
    m_bar = 0.0

    # log stddev of b
    s_bar = np.log(0.1)

    loss_lists_for_all_of_the_sample_sizes = []
    for n_samples in n_samples_list:

        loss_list_for_one_of_the_sample_sizes = []

        for m_tilda in m_tilda_list:

            sum_loss_for_n_samples = 0
            for n in range(0, n_samples):


                if part == 'a':
                    loss = approximate_loss(m_tilda, m_bar, s_tilda, s_bar, x_train_N, y_train_N)
                if part == 'c':
                    loss = approximate_loss_grad(m_tilda, m_bar, s_tilda, s_bar, x_train_N, y_train_N)

                sum_loss_for_n_samples += loss

            sum_loss_for_n_samples_and_m_tilda = sum_loss_for_n_samples / n_samples

            loss_list_for_one_of_the_sample_sizes.append(sum_loss_for_n_samples_and_m_tilda)

        loss_lists_for_all_of_the_sample_sizes.append(loss_list_for_one_of_the_sample_sizes)

    plot_prior_data(m_tilda_list, n_samples_list, loss_lists_for_all_of_the_sample_sizes)


def approximate_loss(m_tilda, m_bar, s_tilda, s_bar, x_train_N, y_train_N):
    bnn = Bnn([Unit()], 0, 1, np.tanh)
    log_p_w = bnn.get_log_p_w()
    log_p_b = bnn.get_log_p_b()
    log_p_w_b = log_p_w + log_p_b
    log_p_y_given_x_w_b = bnn.log_likelihood(y_train_N, x_train_N)
    log_q = bnn.get_log_q_w_b_given_m_s(m_tilda, m_bar, s_tilda, s_bar)
    L = log_p_w_b + log_p_y_given_x_w_b - log_q
    return (L)


approximate_loss_grad = autograd.grad(approximate_loss)

def plot_prior_data(x_grid_G, total_number_samples_list, actual_data):
    fig_h, subplot_grid = plt.subplots(
        nrows=1, ncols=len(total_number_samples_list), sharex=True, sharey=False, squeeze=True, figsize=(10,10))



    for index_col, number_of_samples in enumerate(total_number_samples_list):
        samples = actual_data[index_col]

        samples = np.asarray(samples)
        samples_t = samples.transpose()

        subplot_grid[index_col].plot(x_grid_G, samples_t, linestyle='-.')
        subplot_grid[index_col].set_xlabel('m_tilda')
        subplot_grid[index_col].set_ylabel('#n samples={}'.format(number_of_samples))
        subplot_grid[index_col].set_yticks([])


    plt.show()




problem1(part='a')
#FIXME: why not using the negative?
problem1(part='c')
#why the values are so wild -300000 to + 300000
