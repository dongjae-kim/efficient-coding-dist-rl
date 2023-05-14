#!/usr/bin/env python
# coding: utf-8

# load
import scipy.io as sio
from scipy.optimize import minimize
import numpy as np
import tqdm

np.random.seed(2022)
juiceAmounts = [0.1, 0.3, 1.2, 2.5, 5, 10, 20]


# Fitting functions for sigmoids:


def sigmoid_func_l(x, a, b, c):
    # numerically unstable non log:
    # return b / (1 + np.exp(-(x - c) * a))
    return np.log(b) - np.logaddexp(0, -(x - c) * a)  # log(1+exp(-(x - c) * a))


# first derivative of the sigmoid function
def first_derivative_sigmoid_func(x, a, b, c):
    # numerically unstable:
    # return a * b * np.exp(-a * (x - c)) / (1 + np.exp(-a * (x - c))) ** 2
    return a * b * np.exp(-a * (x - c) - 2 * np.logaddexp(0, -(x - c) * a))


def grad_sigmoid_func(x, a, b, c):
    # now for log:
    g_a = (x - c) * np.exp(-np.logaddexp(0, -(x - c) * a) - ((x - c) * a))
    g_b = np.ones_like(x) / b
    g_c = -a * np.exp(-np.logaddexp(0, -(x - c) * a) - (x - c) * a)
    grad = np.stack([g_a, g_b, g_c])
    return grad


def poisson_lik_l(y, y_pred_l):
    # returns the poisson log-likelihood(s) for data y and prediction y_pred
    # and its derivative for optimization
    y_pred = np.exp(y_pred_l)
    lik = y_pred_l * y - y_pred  # - gammaln(y+1) depends only on y
    lik_d = y - y_pred
    return lik, lik_d


def poisson_lik(y, y_pred):
    # returns the poisson log-likelihood(s) for data y and prediction y_pred
    # and its derivative for optimization
    y_pred_l = np.log(y_pred)
    lik = y_pred_l * y - y_pred  # - gammaln(y+1) depends only on y
    lik_d = y - y_pred
    return lik, lik_d


def poisson_lik_sig(y, x, a, b, c, w=None):
    # poisson likelihood for data y and a sigmoid with parameters a, b, c
    y_pred_l = sigmoid_func_l(x, a, b, c)
    lik, lik_d = poisson_lik_l(y, y_pred_l)
    grad = grad_sigmoid_func(x, a, b, c)
    for i in range(lik.ndim - 1):
        grad = np.expand_dims(grad, -1)
    if w is None:
        lik_sum = np.sum(lik)
        lik_grad = grad * lik_d
    else:
        lik_sum = np.sum(w * lik)
        lik_grad = w * grad * lik_d
    for i in range(lik.ndim):
        lik_grad = np.sum(lik_grad, -1)
    return -lik_sum, -lik_grad


def fit_sigmoid(x, y, x_init=None, w=None):
    if x_init is None:
        x_init = np.array((1, 10, 1))

    def f(par):
        return poisson_lik_sig(y, x, par[0], par[1], par[2], w=w)

    res = minimize(
        f,
        x_init,
        jac=True,
        bounds=(
            [
                (np.finfo(float).eps, 3),
                (np.finfo(float).eps, 100),
                (0.01, 20),  # range of juice rewards
            ]
        ),
    )
    return res.x, res.fun


def check_grad(pars=np.array([1, 10, 1]), delta=0.00001):
    # check gradient:
    def f(par):
        return poisson_lik_sig(y, x, par[0], par[1], par[2], w=None)

    x = np.linspace(-5, 5, 11)
    y = np.random.poisson(np.exp(sigmoid_func_l(x, 1, 10, 1)))
    y0, g = f(pars)
    y1, _ = f(pars + np.array([delta, 0, 0]))
    y2, _ = f(pars + np.array([0, delta, 0]))
    y3, _ = f(pars + np.array([0, 0, delta]))

    print((y1 - y0) / g[0] / delta)
    print((y2 - y0) / g[1] / delta)
    print((y3 - y0) / g[2] / delta)

    y_pred_l = sigmoid_func_l(x, pars[0], pars[1], pars[2])
    lik, lik_d = poisson_lik_l(y, y_pred_l)
    y_pred_l1 = y_pred_l
    y_pred_l1[0] += delta
    lik1, _ = poisson_lik_l(y, y_pred_l)

    print((lik1 - lik) / lik_d[0] / delta)

    a, b, c = pars
    grad = grad_sigmoid_func(x, a, b, c)
    s = sigmoid_func_l(x, a, b, c)
    s1 = sigmoid_func_l(x, a + delta, b, c)
    s2 = sigmoid_func_l(x, a, b + delta, c)
    s3 = sigmoid_func_l(x, a, b, c + delta)
    print((s1 - s) / grad[0] / delta)
    print((s2 - s) / grad[1] / delta)
    print((s3 - s) / grad[2] / delta)


def extract_data(dat, i, juiceAmounts=juiceAmounts):
    dats = dat[0, i][0]
    datas_x = []
    datas_y = []

    for count, j in enumerate(juiceAmounts):
        y_to_extend = np.squeeze(dats[np.where(~np.isnan(dats[:, count])), count])
        x_to_extend = (np.ones(y_to_extend.shape) * j).tolist()
        datas_x.extend(x_to_extend)
        datas_y.extend(y_to_extend)

    x = np.array(datas_x)
    y = np.array(datas_y)
    return x, y


def load_all_data():
    dat = sio.loadmat("measured_neurons/dat_eachneuron.mat")
    dat = dat["dat"]
    data = [extract_data(dat, i) for i in range(dat.shape[1])]
    return data


if __name__ == "__main__":
    # simple fits:
    data = load_all_data()
    initial_guess = [1, 1, 0]
    remove_min = True

    ps = np.zeros((40, 3))
    for i in tqdm.trange(40):
        x, y = data[i]
        if remove_min:
            y -= np.min(y)
        pars, lik = fit_sigmoid(x, y, x_init=initial_guess)
        ps[i, :] = np.array(pars)

    if remove_min:
        sio.savemat("curve_fit_parameters_min.mat", {"ps": ps})
    else:
        sio.savemat("curve_fit_parameters.mat", {"ps": ps})

    # Bootstrapping trials
    num_sim = int(5e3)

    ps = np.zeros((40, num_sim, 3))
    for i in tqdm.trange(40):
        x, y = data[i]
        if remove_min:
            y -= np.min(y)

        for simi in range(num_sim):
            i_sample = np.random.choice(
                np.linspace(0, len(x) - 1, len(x), dtype=np.int16), len(x)
            )
            x_ = x[i_sample]
            y_ = y[i_sample]
            pars, lik = fit_sigmoid(x_, y_, x_init=initial_guess)
            ps[i, simi, :] = np.array(pars)

    if remove_min:
        sio.savemat("curve_fit_bootstrap_min.mat", {"ps": ps})
    else:
        sio.savemat("curve_fit_bootstrap.mat", {"ps": ps})
    # Bootstrapping trials & neurons
    ps = np.zeros((40, num_sim, 3))
    for simi in tqdm.trange(num_sim):
        n_sample = np.random.choice(np.arange(40, dtype=int))  # neuron sample
        for count_n, i in enumerate(n_sample):
            x, y = data[i]
            if remove_min:
                y -= np.min(y)

            i_sample = np.random.choice(
                np.linspace(0, len(x) - 1, len(x), dtype=np.int16), len(x)
            )
            x_ = x[i_sample]
            y_ = y[i_sample]

            pars, lik = fit_sigmoid(x_, y_, x_init=initial_guess)

            ps[count_n, simi, :] = np.array(pars)
            # print('neuron {} simulation {}'.format(count_n, simi))

    if remove_min:
        sio.savemat("curve_fit_bootstrap_neurons_min.mat", {"ps": ps})
    else:
        sio.savemat("curve_fit_bootstrap_neurons.mat", {"ps": ps})
