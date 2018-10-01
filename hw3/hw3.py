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

    n_samples_list = [1, 10]#, 100, 1000]

    # 𝑚̃, mean of 𝑤
    m_tilda_list = np.linspace(-3.0, 5.0, 20)


    # 𝑠̃, log stddev of 𝑤
    s_tilda = np.log(0.1)

    # 𝑚¯, mean of 𝑏
    m_bar = 0.0

    # 𝑠¯, log stddev of 𝑏
    s_bar = np.log(0.1)


    for n_samples in n_samples_list:
        loss_list = []
        for m_tilda in m_tilda_list:
            loss = approximate_loss(m_bar, m_tilda, s_bar, s_tilda, x_train_N, y_train_N)
            loss_list.append(loss)


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

def draw_prior_samples_bnn(m_bar, m_tilda, s_bar, s_tilda, x_train_N, y_train_N, n_samples, x_grid_G):

        sum_losses = 0
        for i in range(0, n_samples):
            sum_losses += approximate_loss(m_bar, m_tilda, s_bar, s_tilda, x_train_N, y_train_N)
        avg_loss = sum_losses / n_samples



def plot_prior_data(x_grid_G, total_number_of_mc_samples_list):
    fig_h, subplot_grid = plt.subplots(
        nrows=0, ncols=len(total_number_of_mc_samples_list), sharex=True, sharey=False, squeeze=True)


    for index_col, number_of_samples in enumerate(total_number_of_mc_samples_list):
        # FIXME: each panel should show 5 samples from the prior
        samples = draw_prior_samples_bnn(number_of_samples, x_grid_G)

        samples = np.asarray(samples)
        samples_t = samples.transpose()

        subplot_grid[0, index_col].plot(x_grid_G, samples_t, linestyle='-.')
        subplot_grid[0, index_col].set_xlabel('m_tilda')
        subplot_grid[0, index_col].set_ylabel('m={}'.format(number_of_samples))
    plt.show()




problem1()
