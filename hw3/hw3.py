# BDL HW3 by Behnam Heydarshahi
# September-October 2018

from __future__ import absolute_import

import numpy as np

np.random.seed(42)

import matplotlib.pyplot as plt


from models.bnn import Bnn
from models.unit import Unit



def problem1():
    x_train_N = np.asarray([-5.0, -2.50, 0.00, 2.50, 5.0])
    y_train_N = np.asarray([-4.91, -2.48, 0.05, 2.61, 5.09])

    n_samples_list = [1, 10, 100, 1000]

    # 𝑚̃, mean of 𝑤
    m_tilda_list = np.linspace(-3.0, 5.0, 20)


    # 𝑠̃, log stddev of 𝑤
    s_tilda = np.log(0.1)

    # 𝑚¯, mean of 𝑏
    m_bar = 0.0

    # 𝑠¯, log stddev of 𝑏
    s_bar = np.log(0.1)

    loss_lists_for_all_of_the_sample_sizes = []
    for n_samples in n_samples_list:

        loss_list_for_one_of_the_sample_sizes = []

        for m_tilda in m_tilda_list:

            sum_loss_for_n_samples = 0
            for n in range(0, n_samples):

                loss = approximate_loss(m_bar, m_tilda, s_bar, s_tilda, x_train_N, y_train_N)
                sum_loss_for_n_samples += loss

            sum_loss_for_n_samples_and_m_tilda = sum_loss_for_n_samples / n_samples

            loss_list_for_one_of_the_sample_sizes.append(sum_loss_for_n_samples_and_m_tilda)

        loss_lists_for_all_of_the_sample_sizes.append(loss_list_for_one_of_the_sample_sizes)


    plot_prior_data(m_tilda_list, n_samples_list, loss_lists_for_all_of_the_sample_sizes)






def approximate_loss(m_bar, m_tilda, s_bar, s_tilda, x_train_N, y_train_N):
    bnn = Bnn([Unit()], 0, 1, np.tanh)
    log_p_w = bnn.get_log_p_w()
    log_p_b = bnn.get_log_p_b()
    log_p_w_b = log_p_w + log_p_b
    log_p_y_given_x_w_b = bnn.log_likelihood(y_train_N, x_train_N)
    log_q = bnn.get_log_q_w_b_given_m_s(m_tilda, s_tilda, m_bar, s_bar)
    # L = E[log_p(y|x, w) + logp(w,b) - logq(w,b|m,s)]
    L = log_p_w_b + log_p_y_given_x_w_b - log_q
    return -1 * L


def plot_prior_data(x_grid_G, total_number_samples_list, actual_data):
    fig_h, subplot_grid = plt.subplots(
        nrows=1, ncols=len(total_number_samples_list), sharex=True, sharey=True, squeeze=True, figsize=(10,10))

    for index_col, number_of_samples in enumerate(total_number_samples_list):
        samples = actual_data[index_col]

        samples = np.asarray(samples)
        samples_t = samples.transpose()


        subplot_grid[index_col].plot(x_grid_G, samples_t, linestyle='-.')
        subplot_grid[index_col].set_xlabel('m_tilda')
        subplot_grid[index_col].set_ylabel('#n samples={}'.format(number_of_samples))
    plt.show()




problem1()
