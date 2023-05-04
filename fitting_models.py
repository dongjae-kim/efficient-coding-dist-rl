import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import poisson, lognorm
import os
import time
import scipy.io as sio
import scipy
from scipy.optimize import minimize, curve_fit

# efficient coding using sigmoid response functions

N_neurons = 39
R_t = 245.41
slope_scale = 5.07

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


def poisson_lik(y, y_pred_l):
    # returns the poisson log-likelihood(s) for data y and prediction y_pred
    # and its derivative for optimization
    y_pred = np.exp(y_pred_l)
    lik = y_pred_l * y - y_pred  # - gammaln(y+1) depends only on y
    lik_d = y - y_pred
    return lik, lik_d


def poisson_lik_sig(y, x, a, b, c, w=None):
    # poisson likelihood for data y and a sigmoid with parameters a, b, c
    y_pred_l = sigmoid_func_l(x, a, b, c)
    lik, lik_d = poisson_lik(y, y_pred_l)
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
                (np.finfo(float).eps, 2),
                (np.finfo(float).eps, 100),
                (np.finfo(float).eps, 30),
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
    lik, lik_d = poisson_lik(y, y_pred_l)
    y_pred_l1 = y_pred_l
    y_pred_l1[0] += delta
    lik1, _ = poisson_lik(y, y_pred_l)

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


class value_efficient_coding_moment:
    def __init__(
        self,
        N_neurons=18,
        R_t=247.0690,
        X_OPT_ALPH=1.0,
        slope_scale=slope_scale,
        simpler=False,
    ):
        # real data prior
        self.offset = 0
        # it is borrowed from the original data of Dabney's
        self.juice_magnitudes = np.array([0.1, 0.3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array(
            [
                0.06612594,
                0.09090909,
                0.14847358,
                0.15489467,
                0.31159175,
                0.1509519,
                0.07705306,
            ]
        )
        self.juice_prob /= np.sum(self.juice_prob)

        p_thresh = (2 * np.arange(N_neurons) + 1) / N_neurons / 2

        if simpler:  # to boost computation
            self.x = np.linspace(np.finfo(float).eps, 30, num=int(1e3))
            self.x_inf = np.linspace(np.finfo(float).eps, 300, num=int(1e4))
        else:
            self.x = np.linspace(np.finfo(float).eps, 30, num=int(1e4))
            self.x_inf = np.linspace(np.finfo(float).eps, 300, num=int(1e5))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        # np.linspace(-50, 50, num=int(1e4))
        self.x_log_inf = np.log(self.x_inf)

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 21]

        self.p_prior = lognorm.pdf(self.x, s=0.71, scale=np.exp(1.289))
        self.p_prior_inf = lognorm.pdf(self.x_inf, s=0.71, scale=np.exp(1.289))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / np.sum(self.p_prior_inf * self._x_gap)

        # pseudo p-prior to make the sum of the p-prior in the range can be 1
        self.p_prior_pseudo = []
        ppp_cumsum = np.cumsum(self.p_prior_inf * self._x_gap)
        ppp_cumsum /= ppp_cumsum[-1]  # Offset
        self.p_prior_pseudo.append(ppp_cumsum[0])
        for i in range(len(ppp_cumsum) - 1):
            self.p_prior_pseudo.append(
                (ppp_cumsum[i + 1] - ppp_cumsum[i]) / self._x_gap
            )
        self.p_prior_pseudo = np.array(self.p_prior_pseudo)

        # since we posit a distribution ranged in [0,20] (mostly) we hypothesized
        # that integral from -inf to +inf is same
        # as the integral from 0 to 20 in this toy example.
        # From now on, we just calculated cumulative distribution using
        # self.x, which ranged from 0 to 20.

        # a prototype sigmoidal response curve
        self.h_s = lambda x: 1 / (1 + np.exp(x))

        # number of neurons
        self.N = N_neurons

        # total population response: mean of R spikes
        self.R = R_t

        # p_prior_sum = self.p_prior/np.sum(self.p_prior)
        # self.cum_P = np.cumsum(p_prior_sum)

        # to prevent 0 on denominator in self.g
        p_prior_sum = self.p_prior / np.sum(self.p_prior)
        self.cum_P = np.cumsum(p_prior_sum)  # - 1e-3  # for approximation
        self.cum_P /= 1 + 1e-3

        # p_prior_inf_sum = self.p_prior_inf/np.sum(self.p_prior_inf)
        p_prior_inf_sum = self.p_prior_inf / np.sum(self.p_prior_inf)
        self.cum_P_pseudo = np.cumsum(p_prior_inf_sum)  # - 1e-5  # for approximation
        self.cum_P_pseudo /= 1 + 1e-3

        norm_d = self.p_prior / (1 - self.cum_P) ** (1 - X_OPT_ALPH)
        NRMLZR = np.sum(norm_d * self._x_gap)
        norm_d = norm_d / NRMLZR

        cum_norm_D = np.cumsum(self.N * norm_d * self._x_gap)
        cum_norm_Dp = np.cumsum(self.N * norm_d * self._x_gap) / cum_norm_D[-1]

        thresh_ = np.interp(p_thresh, cum_norm_Dp, self.x)
        quant_ = np.interp(thresh_, self.x, cum_norm_Dp)

        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g = 1 / ((1 - self.cum_P) ** X_OPT_ALPH)
        # norm_g /= NRMLZR
        norm_g /= self.N
        norm_g *= self.R

        norm_d_pseudo = self.p_prior_pseudo / (1 - self.cum_P_pseudo) ** (
            1 - X_OPT_ALPH
        )
        NRMLZR_pseudo = np.sum(norm_d_pseudo * self._x_gap)
        norm_d_pseudo = norm_d_pseudo / NRMLZR_pseudo

        cum_norm_D_pseudo = np.cumsum(self.N * norm_d_pseudo * self._x_gap)
        cum_norm_D_pseudop = (
            np.cumsum(self.N * norm_d_pseudo * self._x_gap) / cum_norm_D_pseudo[-1]
        )

        thresh_pseudo_ = np.interp(p_thresh, cum_norm_D_pseudop, self.x_inf)
        quant_pseudo_ = np.interp(thresh_pseudo_, self.x_inf, cum_norm_D_pseudop)

        norm_g_pseudo = 1 / ((1 - self.cum_P_pseudo) ** X_OPT_ALPH)
        norm_g_pseudo /= self.N
        norm_g_pseudo *= self.R

        # find each neuron's location preferred response of each neuron.
        # It is x=0 in the prototype sigmoid function (where y=0.5)
        self.sn = thresh_
        self.sn_pseudo = thresh_pseudo_
        self.gsn = []
        self.gsn_pseudo = []

        # each neurons response function
        self.neurons_ = []  # self.N number of neurons
        self.neurons_pseudo_ = []  # self.N number of neurons

        for i in range(N_neurons):
            g_sn = norm_g[np.argmin(np.abs(self.x - self.sn[i]))]

            a = slope_scale * quant_[i]
            b = slope_scale * (1 - quant_[i])
            self.neurons_.append(g_sn * scipy.special.betainc(a, b, cum_norm_Dp))
            self.gsn.append(g_sn)

            g_sn = norm_g_pseudo[np.argmin(np.abs(self.x_inf - self.sn_pseudo[i]))]

            a = slope_scale * quant_pseudo_[i]
            b = slope_scale * (1 - quant_pseudo_[i])
            self.neurons_pseudo_.append(
                g_sn * scipy.special.betainc(a, b, cum_norm_D_pseudop)
            )
            self.gsn_pseudo.append(g_sn)

        # normalize afterward
        NRMLZR_G = self.R / np.sum(np.array(self.neurons_) * self.p_prior * self._x_gap)
        # neurons_arr=np.array(self.neurons_)*NRMLZR_G
        for i in range(len(self.neurons_)):
            self.neurons_[i] *= NRMLZR_G
            self.gsn[i] *= NRMLZR_G
        NRMLZR_G_pseudo = self.R / np.sum(
            np.array(self.neurons_pseudo_) * self.p_prior_pseudo * self._x_gap
        )
        # neurons_arr=np.array(self.neurons_)*NRMLZR_G
        for i in range(len(self.neurons_pseudo_)):
            self.neurons_pseudo_[i] *= NRMLZR_G_pseudo
            self.gsn_pseudo[i] *= NRMLZR_G_pseudo

        self.d_x = norm_d
        self.d_x_pseudo = norm_d_pseudo
        self.g_x = norm_g * NRMLZR_G
        self.g_x_pseudo = norm_g_pseudo * NRMLZR_G_pseudo
        self.initial_dabney()

    def get_quantiles_RPs(self, quantiles):
        P_PRIOR = np.cumsum(self.p_prior_inf * self._x_gap)
        RPs = []
        for i in range(len(quantiles)):
            indx = np.argmin(abs(P_PRIOR - quantiles[i]))
            RPs.append(self.x_inf[indx])
        return RPs

    def initial_dabney(self):
        fig5 = sio.loadmat("./measured_neurons/dabney_matlab/dabney_fit.mat")
        fig5_betas = sio.loadmat(
            "./measured_neurons/dabney_matlab/dabney_utility_fit.mat"
        )
        scaleFactNeg_all = fig5["scaleFactNeg_all"][:, 0]
        scaleFactPos_all = fig5["scaleFactPos_all"][:, 0]
        asymM_all = fig5["asymM_all"][:, 0]

        def ZC_estimator(x):
            return fig5_betas["betas"][0, 0] + fig5_betas["betas"][1, 0] * x

        idx_to_maintain = np.where((scaleFactNeg_all * scaleFactPos_all) > 0)[0]
        asymM_all = asymM_all[idx_to_maintain]
        idx_sorted = np.argsort(asymM_all)
        asymM_all = asymM_all[idx_sorted]
        estimated_ = np.array(self.get_quantiles_RPs(asymM_all))
        zero_crossings_ = fig5["zeroCrossings_all"][:, 0]
        zero_crossings_ = zero_crossings_[idx_to_maintain]
        zero_crossings_ = zero_crossings_[idx_sorted]
        zero_crossings_estimated = ZC_estimator(zero_crossings_)

        # ver 2
        dir_measured_neurons = "measured_neurons/"
        NDAT = sio.loadmat(dir_measured_neurons + "data_max.mat")["dat"]
        zero_crossings_estimated = NDAT["ZC"][0, 0].squeeze()

        self.Dabneys = [estimated_, zero_crossings_estimated]

    def plot_approximate_kinky(self, r_star=0.02, name="gamma"):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        for i in range(self.N):  # excluded the last one since it is noisy
            (
                (E_hn_prime_lower, E_hn_prime_higher, quantile),
                (x, y),
                theta_x,
            ) = self.hn_approximate(i, r_star)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))

        if name == "gamma":
            plt.ylim((0, 0.10))
            plt.xlim((0.5, 2.5))
        elif name == "normal":
            plt.ylim((0, 0.10))
            plt.xlim((3, 6))
        plt.title("Approximated")
        plt.show()
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(name + "/" + "Approximated.png")
        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("quantile for each neuron")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(name + "/" + "quantile for each neuron.png")

        return quantiles

    def hn_approximate(self, n, r_star=0.02):
        hn = self.neurons_[n]
        theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
        E_hn_prime_higher = np.sum(inner_sigma_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (
            (
                E_hn_prime_lower,
                E_hn_prime_higher,
                E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower),
            ),
            (self.x, np.array(out_)),
            theta_x,
        )

    def plot_approximate_kinky_normalized(self, r_star=0.02, name="gamma"):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        theta_s = []
        for i in range(self.N - 1):  # excluded the last one since it is noisy
            (
                (E_hn_prime_lower, E_hn_prime_higher, quantile),
                (x, y),
                theta_x,
            ) = self.hn_approximate_normalized(i, r_star)
            theta_s.append(theta_x)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))

        if name == "gamma":
            plt.ylim((0, 0.1))
            plt.xlim((0, 4))
        elif name == "normal":
            plt.ylim((-0.0, 0.20))
            plt.xlim((3, 6.5))
        plt.title("Approximated")
        plt.show()
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(name + "/" + "Approximated normalized.png")
        plt.figure()
        xlocs = np.linspace(1, self.N - 1, self.N - 1)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("quantile for each neuron")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(name + "/" + "expectile for each neuron normalized.png")

        return quantiles

    def hn_approximate_normalized(self, n, r_star=0.02):
        hn = self.neurons_[n]
        theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap)
            denominator_hn_prime.append(self.p_prior[i] * self.x[i] * self._x_gap)
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap)
            denominator_hn_prime.append(self.p_prior[i] * self.x[i] * self._x_gap)
        E_hn_prime_higher = np.sum(inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (
            (
                E_hn_prime_lower,
                E_hn_prime_higher,
                E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower),
            ),
            (self.x, np.array(out_)),
            theta_x,
        )

    def plot_approximate_kinky_true(self, r_star=0.02, name="gamma"):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        theta_s = []
        alpha_s = []
        x_s = []
        y_s = []
        for i in range(self.N):  # excluded the last one since it is noisy
            (
                (E_hn_prime_lower, E_hn_prime_higher, quantile),
                (x, y),
                theta_x,
            ) = self.hn_approximate_true(i, r_star)
            theta_s.append(theta_x)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))
            alpha_s.append([E_hn_prime_lower, E_hn_prime_higher])
            x_s.append(x)
            y_s.append(y)

        if name == "gamma":
            plt.ylim((0.0, 0.15))
            plt.xlim((0, 8))
        elif name == "normal":
            plt.ylim((0, 0.10))
            plt.xlim((2, 7))
        else:
            print("nothing")
        plt.title("Approximated")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "Approximated true.png")

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("quantile for each neuron")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "expectile for each neuron true.png")

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(self.sn_pseudo))
        for i, v in enumerate(np.array(self.sn_pseudo)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("preferred reward")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "preferred reward.png")

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(theta_s))
        for i, v in enumerate(np.array(theta_s)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("reversal points")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "reversal points.png")

        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples_fitting_only(
        self, samples_idx, neurons, name, r_star_param=0.02
    ):
        # cal r-star
        if r_star_param < 1:
            res_max = []
            res_min = []
            for i in range(len(self.neurons_)):
                res_max.append(np.max(self.neurons_[i]))
                res_min.append(np.min(self.neurons_[i]))
            r_star = np.min(res_max) * r_star_param
        else:
            r_stars = np.copy(self.gsn_pseudo) / 2

        def func(rp, offset):
            return lambda x, a: a * (x - rp) + offset

        # get threshold for each neuron
        thresholds = []
        thresholds_idx = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            hn = neurons[i]
            ind = np.argmin(abs(hn - r_star))
            thresholds.append(self.x[ind])
            thresholds_idx.append(ind)

        X_minus = []
        X_plus = []
        R_minus = []
        R_plus = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            X_minus_ = []
            X_plus_ = []
            R_minus_ = []
            R_plus_ = []
            for xi in samples_idx:
                if xi > thresholds_idx[i]:
                    X_plus_.append(self.x[xi])
                    R_plus_.append(neurons[i][xi])
                else:
                    X_minus_.append(self.x[xi])
                    R_minus_.append(neurons[i][xi])
            X_minus.append(X_minus_)
            R_minus.append(R_minus_)
            X_plus.append(X_plus_)
            R_plus.append(R_plus_)

        alpha_plus = []
        alpha_minus = []
        alphas = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            idx_sort = np.argsort(X_minus[i])
            popt_minus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_minus[i])[idx_sort],
                np.array(R_minus[i])[idx_sort],
            )
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_plus[i])[idx_sort],
                np.array(R_plus[i])[idx_sort],
            )
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (
            np.divide(np.array(alpha_s), np.sum(np.array(alpha_s), 1).reshape(-1, 1))
        )[:, 1]
        x_s = []
        y_s = []
        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples_fitting_only_raw_rstar(
        self, samples_idx, neurons, r_star_param
    ):
        # cal r-star
        # r_star = r_star_param
        # cal r-star
        # r_star = r_star_param

        def func(rp, offset):
            return lambda x, a: a * (x - rp) + offset

        # get threshold for each neuron
        thresholds = []
        thresholds_idx = []
        for i in range(self.N):
            r_star = r_star_param[i]
            hn = neurons[i]
            ind = np.argmin(abs(hn - r_star))
            thresholds.append(self.x[ind])
            thresholds_idx.append(ind)

        X_minus = []
        X_plus = []
        R_minus = []
        R_plus = []
        for i in range(self.N):
            r_star = r_star_param[i]
            X_minus_ = []
            X_plus_ = []
            R_minus_ = []
            R_plus_ = []
            for xi in samples_idx:
                if xi > thresholds_idx[i]:
                    X_plus_.append(self.x[xi])
                    R_plus_.append(neurons[i][xi])
                else:
                    X_minus_.append(self.x[xi])
                    R_minus_.append(neurons[i][xi])
            X_minus.append(X_minus_)
            R_minus.append(R_minus_)
            X_plus.append(X_plus_)
            R_plus.append(R_plus_)

        alpha_plus = []
        alpha_minus = []
        alphas = []
        for i in range(self.N):
            r_star = r_star_param[i]

            idx_sort = np.argsort(X_minus[i])
            popt_minus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_minus[i])[idx_sort],
                np.array(R_minus[i])[idx_sort],
            )
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_plus[i])[idx_sort],
                np.array(R_plus[i])[idx_sort],
            )
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

            # ind_ = thresholds_idx[i]
            # x1_ = np.sum( self.x[:ind_]*self.p_prior[:ind_]*self._x_gap )
            # y1_ = np.sum( self.p_prior[:ind_]*self.neurons_[i][:ind_]*self._x_gap)
            # alpha_minus.append(y1_/x1_)
            # x2_ = np.sum( self.x[ind_:]*self.p_prior[ind_:]*self._x_gap )
            # y2_ = np.sum( self.p_prior[ind_:]*self.neurons_[i][ind_:]*self._x_gap)
            # alpha_plus.append(y2_/x2_)
            #
            # alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (
            np.divide(np.array(alpha_s), np.sum(np.array(alpha_s), 1).reshape(-1, 1))
        )[:, 1]
        x_s = []
        y_s = []
        return quantiles, theta_s, alpha_s, x_s, y_s

    def get_thresholds(self, r_star):
        # get threshold for each neuron
        thresholds = []
        thresholds_idx = []
        for i in range(self.N):
            hn = self.neurons_[i]
            ind = np.argmin(abs(hn - r_star))
            thresholds_idx.append(ind)
            thresholds.append(np.interp(r_star, self.neurons_[i], self.x))
        return thresholds, thresholds_idx

    def plot_approximate_kinky_fromsim_fitting_only_raw_rstar(
        self, r_star, num_samples=int(1e5)
    ):
        def func(rp, offset):
            return lambda x, a: a * (x - rp) + offset

        thresholds, thresholds_idx = self.get_thresholds(r_star)

        alpha_plus = []
        alpha_minus = []
        alphas = []
        num_samples_yours = [num_samples] * self.N
        for i in range(self.N):
            # check it before
            p = self.p_prior[: thresholds_idx[i]] + np.finfo(float).eps
            rand_neg = np.random.choice(
                np.arange(0, thresholds_idx[i]),
                p=p / np.sum(p),
                size=int(
                    np.ceil(num_samples_yours[i] * self.cum_P_pseudo[thresholds_idx[i]])
                ),
            )
            sample_neg = self.x[rand_neg]
            R_neg = self.neurons_[i][rand_neg]
            p = self.p_prior[thresholds_idx[i] :] + np.finfo(float).eps
            rand_pos = np.random.choice(
                np.arange(thresholds_idx[i], len(self.x)),
                p=p / np.sum(p),
                size=int(
                    np.ceil(
                        num_samples_yours[i]
                        * (1 - self.cum_P_pseudo[thresholds_idx[i]])
                    )
                ),
            )
            sample_pos = self.x[rand_pos]
            R_pos = self.neurons_[i][rand_pos]
            neg_idx = np.argsort(sample_neg)
            popt_v1_minus, _ = curve_fit(
                func(self.x[thresholds_idx[i]], r_star),
                np.array(sample_neg)[neg_idx],
                np.array(R_neg)[neg_idx],
                sigma=np.sqrt(np.array(R_neg)[neg_idx] + np.finfo(float).eps),
            )
            pos_idx = np.argsort(sample_pos)
            popt_v1_plus, _ = curve_fit(
                func(self.x[thresholds_idx[i]], r_star),
                np.array(sample_pos)[pos_idx],
                np.array(R_pos)[pos_idx],
                sigma=np.sqrt(np.array(R_pos)[pos_idx] + np.finfo(float).eps),
            )

            alpha_minus.append(*popt_v1_minus)
            alpha_plus.append(*popt_v1_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (
            np.divide(np.array(alpha_s), np.sum(np.array(alpha_s), 1).reshape(-1, 1))
        )[:, 1]
        x_s = []
        y_s = []
        return True, quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples_fitting_only_rtimesg(
        self, samples_idx, neurons, name, gx, r_star_param=0.02
    ):
        r_star_param *= np.mean(gx[samples_idx])
        # cal r-starr
        if r_star_param < 1:
            res_max = []
            res_min = []
            for i in range(len(self.neurons_)):
                res_max.append(np.max(self.neurons_[i]))
                res_min.append(np.min(self.neurons_[i]))
            r_star = np.min(res_max) * r_star_param
        else:
            r_stars = np.copy(self.gsn_pseudo) / 2

        def func(rp, offset):
            return lambda x, a: a * (x - rp) + offset

        # get threshold for each neuron
        thresholds = []
        thresholds_idx = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            hn = neurons[i]
            ind = np.argmin(abs(hn - r_star))
            thresholds.append(self.x[ind])
            thresholds_idx.append(ind)

        X_minus = []
        X_plus = []
        R_minus = []
        R_plus = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            X_minus_ = []
            X_plus_ = []
            R_minus_ = []
            R_plus_ = []
            for xi in samples_idx:
                if xi > thresholds_idx[i]:
                    X_plus_.append(self.x[xi])
                    R_plus_.append(neurons[i][xi])
                else:
                    X_minus_.append(self.x[xi])
                    R_minus_.append(neurons[i][xi])
            X_minus.append(X_minus_)
            R_minus.append(R_minus_)
            X_plus.append(X_plus_)
            R_plus.append(R_plus_)

        alpha_plus = []
        alpha_minus = []
        alphas = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            idx_sort = np.argsort(X_minus[i])
            popt_minus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_minus[i])[idx_sort],
                np.array(R_minus[i])[idx_sort],
            )
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_plus[i])[idx_sort],
                np.array(R_plus[i])[idx_sort],
            )
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (
            np.divide(np.array(alpha_s), np.sum(np.array(alpha_s), 1).reshape(-1, 1))
        )[:, 1]
        x_s = []
        y_s = []
        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples(
        self, samples_idx, neurons, name, r_star_param=0.02
    ):
        # cal r-star
        if r_star_param < 1:
            res_max = []
            res_min = []
            for i in range(len(self.neurons_)):
                res_max.append(np.max(self.neurons_[i]))
                res_min.append(np.min(self.neurons_[i]))
            r_star = np.min(res_max) * 0.8
        else:
            r_stars = np.copy(self.gsn_pseudo) / 2

        def func(rp, offset):
            return lambda x, a: a * (x - rp) + offset

        # get threshold for each neuron
        thresholds = []
        thresholds_idx = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            hn = neurons[i]
            ind = np.argmin(abs(hn - r_star))
            thresholds.append(self.x[ind])
            thresholds_idx.append(ind)

        X_minus = []
        X_plus = []
        R_minus = []
        R_plus = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            X_minus_ = []
            X_plus_ = []
            R_minus_ = []
            R_plus_ = []
            for xi in samples_idx:
                if xi > thresholds_idx[i]:
                    X_plus_.append(self.x[xi])
                    R_plus_.append(neurons[i][xi])
                else:
                    X_minus_.append(self.x[xi])
                    R_minus_.append(neurons[i][xi])
            X_minus.append(X_minus_)
            R_minus.append(R_minus_)
            X_plus.append(X_plus_)
            R_plus.append(R_plus_)

        alpha_plus = []
        alpha_minus = []
        alphas = []
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            idx_sort = np.argsort(X_minus[i])
            popt_minus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_minus[i])[idx_sort],
                np.array(R_minus[i])[idx_sort],
            )
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(
                func(thresholds[i], r_star),
                np.array(X_plus[i])[idx_sort],
                np.array(R_plus[i])[idx_sort],
            )
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (
            np.divide(np.array(alpha_s), np.sum(np.array(alpha_s), 1).reshape(-1, 1))
        )[:, 1]
        x_s = []
        y_s = []

        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        for i in range(self.N):
            if r_star_param >= 1:
                r_star = r_stars[i]
            x_ = []
            y_ = []
            for xi in range(len(self.x)):
                x_.append(self.x[xi])
                if xi > thresholds_idx[i]:
                    y_.append((self.x[xi] - thresholds[i]) * alpha_s[i][1] + r_star)
                else:
                    y_.append((self.x[xi] - thresholds[i]) * alpha_s[i][0] + r_star)

            plt.plot(x_, y_, color=str(colors[i]))
            x_s.append(x_)
            y_s.append(y_)

        plt.title("Approximated")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(
            "./" + name + "/" + "Approximated true " + str(r_star_param) + " .png"
        )

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("quantile for each neuron")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(
            "./"
            + name
            + "/"
            + "expectile for each neuron true "
            + str(r_star_param)
            + ".png"
        )

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(self.sn_pseudo))
        for i, v in enumerate(np.array(self.sn_pseudo)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("preferred reward")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(
            "./" + name + "/" + "preferred reward " + str(r_star_param) + ".png"
        )

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(theta_s))
        for i, v in enumerate(np.array(theta_s)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("reversal points")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "reversal points " + str(r_star_param) + ".png")

        return quantiles, theta_s, alpha_s, x_s, y_s

    def hn_approximate_true(self, n, r_star=0.01):
        hn = self.neurons_[n]

        ind = np.squeeze(np.where((hn - r_star) > 0))
        # theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta_x = self.x[ind[0]]

        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap
            )  # this will be y value
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap
            )  # this will be x value
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i]) * self._x_gap)
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap
            )  # this will be x value
        E_hn_prime_higher = np.sum(inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # # denominator
        # denominator_hn_prime = []
        # for i in range( len(self.x) - 1):
        #     denominator_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
        #
        # #
        # E_hn_prime_lower = E_hn_prime_lower/ np.sum(denominator_hn_prime)
        # E_hn_prime_higher = E_hn_prime_higher/ np.sum(denominator_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (
            (
                E_hn_prime_lower,
                E_hn_prime_higher,
                E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower),
            ),
            (self.x, np.array(out_)),
            theta_x,
        )

    def plot_approximate_kinky_true_wo_prior(self, r_star=0.02, name="gamma"):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        theta_s = []
        for i in range(self.N):  # excluded the last one since it is noisy
            (
                (E_hn_prime_lower, E_hn_prime_higher, quantile),
                (x, y),
                theta_x,
            ) = self.hn_approximate_true_wo_prior(i, r_star)
            theta_s.append(theta_x)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))

        if name == "gamma":
            plt.ylim((0.01, 0.02))
            plt.xlim((0, 4))
        elif name == "normal":
            plt.ylim((0, 0.10))
            plt.xlim((2, 7))
        plt.title("Approximated")
        plt.show()
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(name + "/" + "Approximated true wo prior.png")
        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("quantile for each neuron")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(name + "/" + "quantile for each neuron true wo prior.png")

        return quantiles

    def hn_approximate_true_wo_prior(self, n, r_star=0.02):
        hn = self.neurons_[n]
        theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap
            )  # this will be y value
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i]) * self._x_gap)
        E_hn_prime_higher = np.sum(inner_sigma_hn_prime)

        # # denominator
        # denominator_hn_prime = []
        # for i in range( len(self.x) - 1):
        #     denominator_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
        #
        # #
        # E_hn_prime_lower = E_hn_prime_lower/ np.sum(denominator_hn_prime)
        # E_hn_prime_higher = E_hn_prime_higher/ np.sum(denominator_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (
            (
                E_hn_prime_lower,
                E_hn_prime_higher,
                E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower),
            ),
            (self.x, np.array(out_)),
            theta_x,
        )

    def plot_neurons(self, name="gamma"):
        # plot neurons response functions
        colors = np.linspace(0, 0.7, self.N)
        plt.figure()
        ymax = []
        for i in range(self.N):
            plt.plot(self.x, self.neurons_[i], color=str(colors[i]))
            ymax.append(self.neurons_[i][1499])
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title("Response functions of {0} neurons".format(self.N))
        if not os.path.exists("./" + name + "/"):
            os.makedirs("./" + name + "/")
        plt.savefig(
            "./"
            + name
            + "/"
            + "Response functions of {0} neurons 300.png".format(self.N)
        )
        plt.xlim([0, 45])
        plt.ylim([0, np.max(ymax)])
        plt.savefig(
            "./" + name + "/" + "Response functions of {0} neurons.png".format(self.N)
        )

    def plot_neurons_pseudo(self, name="gamma"):
        # plot neurons response functions
        colors = np.linspace(0, 0.7, self.N)
        plt.figure()
        for i in range(self.N):
            plt.plot(self.x, self.neurons_pseudo_[i], color=str(colors[i]))
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title("Response functions of {0} neurons".format(self.N))
        plt.show()
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig(
            name + "/" + "Response functions of {0} neurons pseudo.png".format(self.N)
        )

    def plot_others(self, name="gamma"):
        plt.figure()
        plt.title("Prior distribution")
        plt.plot(self.x, self.p_prior)
        if not os.path.exists("./" + name + "/"):
            os.makedirs("./" + name + "/")
        plt.savefig("./" + name + "/" + "Prior distribution 300.png")
        plt.figure()
        plt.title("Density function")
        plt.plot(self.x, self.d_x_pseudo)
        plt.savefig("./" + name + "/" + "Density function 300.png")
        plt.figure()
        plt.title("Gain function")
        plt.plot(self.x, self.g_x_pseudo)
        plt.savefig("./" + name + "/" + "Gain function 300.png")

        # plt.figure()
        # plt.title('Prior distribution')
        # plt.plot(self.x[:ind30], self.p_prior[:ind30])
        # if not os.path.exists('./' + name + '/'):
        #     os.makedirs('./' + name + '/')
        # plt.savefig('./' + name + '/' + 'Prior distribution.png')
        # plt.figure()
        # plt.title('Density function')
        # plt.plot(self.x[:ind30], self.d_x[:ind30])
        # plt.savefig('./' + name + '/' + 'Density function.png')
        # plt.figure()
        # plt.title('Gain function')
        # plt.plot(self.x[:ind30], self.g_x[:ind30])
        # plt.savefig('./' + name + '/' + 'Gain function.png')

    def replace_with_pseudo(self):
        self.sn = self.sn_pseudo
        self.neurons_ = self.neurons_pseudo_
        self.x = self.x_inf
        self.p_prior = self.p_prior_inf
        # self.d_x = self.d_x_pseudo
        # self.g_x = self.g_x_pseudo

    def plot_approximate_variable_rstar(
        self, xy_to_plots, quantiles, theta_s, name="real_bigger"
    ):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        for i in range(self.N):  # excluded the last one since it is noisy
            (x, y) = xy_to_plots[i]
            plt.plot(x, y, color=str(colors[i]))

        plt.title("Approximated")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "Approximated true.png")

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("quantile for each neuron")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "expectile for each neuron true.png")

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(self.sn_pseudo))
        for i, v in enumerate(np.array(self.sn_pseudo)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("preferred reward")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "preferred reward.png")

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(theta_s))
        for i, v in enumerate(np.array(theta_s)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title("reversal points")
        if not os.path.exists(name + "/"):
            os.makedirs(name + "/")
        plt.savefig("./" + name + "/" + "reversal points.png")

        return quantiles, theta_s

    def simul_n_th_neuron(self, n):
        hn = self.neurons_[n]
        # hn = self.neurons_pseudo_[n]

        alpha_minus = []
        alpha_plus = []
        quantile = []
        xy_to_plot = []
        theta_xs = []

        for ind in range(hn.shape[0]):
            theta_x = self.x[ind]
            # theta_x = self.x_inf[ind]
            theta = ind
            r_star = hn[ind]

            # lower than theta = hn^(-1)(r_star)
            inner_sigma_hn_prime = []
            denominator_hn_prime = []
            for i in range(theta):
                inner_sigma_hn_prime.append(
                    self.p_prior[i] * (hn[i]) * self._x_gap
                )  # this will be y value
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior[i] * self.x[i] * self._x_gap
                )  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_lower = np.nan
            else:
                E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / np.sum(
                    denominator_hn_prime
                )

            # higher than theta
            inner_sigma_hn_prime = []
            denominator_hn_prime = []
            for i in range(theta, len(self.x) - 1):
                inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i]) * self._x_gap)
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior[i] * self.x[i] * self._x_gap
                )  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_higher = np.nan
            else:
                E_hn_prime_higher = np.sum(inner_sigma_hn_prime) / np.sum(
                    denominator_hn_prime
                )

            # plot it
            out_ = []
            for i in range(len(self.x)):
                if i < theta:
                    out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
                else:
                    out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

            alpha_minus.append(E_hn_prime_lower)
            alpha_plus.append(E_hn_prime_higher)
            quantile.append(E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower))
            # xy_to_plot.append((self.x, np.array(out_)))
            xy_to_plot.append((self.x, np.array(out_)))
            theta_xs.append(theta_x)

        return alpha_minus, alpha_plus, quantile, xy_to_plot, theta_xs

    def simul_n_th_neuron_pseudo(self, n):
        hn = self.neurons_pseudo_[n]

        alpha_minus = []
        alpha_plus = []
        quantile = []
        xy_to_plot = []
        theta_xs = []

        for ind in range(hn.shape[0]):
            theta_x = self.x_inf[ind]
            theta = ind
            r_star = hn[ind]

            # lower than theta = hn^(-1)(r_star)
            inner_sigma_hn_prime = []
            denominator_hn_prime = []
            for i in range(theta):
                inner_sigma_hn_prime.append(
                    self.p_prior_inf[i] * (hn[i]) * self._x_gap
                )  # this will be y value
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior_inf[i] * self.x_inf[i] * self._x_gap
                )  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_lower = np.nan
            else:
                E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / np.sum(
                    denominator_hn_prime
                )

            # higher than theta
            inner_sigma_hn_prime = []
            denominator_hn_prime = []
            for i in range(theta, len(self.x_inf) - 1):
                inner_sigma_hn_prime.append(self.p_prior_inf[i] * (hn[i]) * self._x_gap)
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior_inf[i] * self.x_inf[i] * self._x_gap
                )  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_higher = np.nan
            else:
                E_hn_prime_higher = np.sum(inner_sigma_hn_prime) / np.sum(
                    denominator_hn_prime
                )

            # plot it
            out_ = []
            # for i in range(len(self.x)):
            #     if i < theta:
            #         out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            #     else:
            #         out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)
            for i in range(len(self.x_inf)):
                if i < theta:
                    out_.append(E_hn_prime_lower * (self.x_inf[i] - theta_x) + r_star)
                else:
                    out_.append(E_hn_prime_higher * (self.x_inf[i] - theta_x) + r_star)

            alpha_minus.append(E_hn_prime_lower)
            alpha_plus.append(E_hn_prime_higher)
            quantile.append(E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower))
            # xy_to_plot.append((self.x, np.array(out_)))
            xy_to_plot.append((self.x_inf, np.array(out_)))
            theta_xs.append(theta_x)

        return alpha_minus, alpha_plus, quantile, xy_to_plot, theta_xs

    def hn_approximate_rstar(self, n, r_star=0.01):
        hn = self.neurons_[n]

        ind = np.squeeze(np.where((hn - r_star) > 0))
        # theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta_x = self.x[ind[0]]

        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap
            )  # this will be y value
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap
            )  # this will be x value
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i]) * self._x_gap)
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap
            )  # this will be x value
        E_hn_prime_higher = np.sum(inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (
            (
                E_hn_prime_lower,
                E_hn_prime_higher,
                E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower),
            ),
            (self.x, np.array(out_)),
            theta_x,
        )

    def gen_simulation_data(self, num_samples=100):
        # sampling values
        sample_x = []
        sample_x_idx = []
        num_samples = num_samples
        for s in range(num_samples):
            sample_x.append(
                np.random.choice(self.x, p=self.p_prior / np.sum(self.p_prior))
            )
            sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])
        self.sample_x = sample_x
        self.sample_x_idx = sample_x_idx

        return self.sample_x, self.sample_x_idx

    def gen_response_poisson(self, neuron, num_samples=1000, num_samples_pp=1):
        # sample responses of neurons
        # num_neurons * num_samples
        # then fitlm as they did.

        sample_x = []
        sample_x_idx = []
        response = []
        for s_id in range(num_samples):
            sample_x.append(
                np.random.choice(self.x, p=self.p_prior / np.sum(self.p_prior))
            )
            sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])
            response_ = []
            for n_id in range(self.N):
                if np.isinf(num_samples_pp):
                    # get respones if you say infinite number of samples
                    response_.append(neuron[n_id][sample_x_idx[-1]])
                else:
                    for pp_id in range(num_samples_pp):
                        mu = neuron[n_id][sample_x_idx[-1]]
                        x = np.arange(poisson.ppf(1e-5, mu), poisson.ppf(1 - 1e-5, mu))
                        y = poisson.pmf(x, mu)
                        y /= np.sum(y)
                        response_.append(np.random.choice(x, p=y))
            response.append(response_)

        self.sample_x = sample_x
        self.sample_x_idx = sample_x_idx
        return self.sample_x, self.sample_x_idx, np.array(response)

    def gen_samples(self, num_samples=1000):
        # sample responses of neurons
        # num_neurons * num_samples
        # then fitlm as they did.

        sample_x = []
        sample_x_idx = []
        for s_id in range(num_samples):
            sample_x.append(
                np.random.choice(self.x, p=self.p_prior / np.sum(self.p_prior))
            )
            sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])

        self.sample_x = sample_x
        self.sample_x_idx = sample_x_idx
        return self.sample_x, self.sample_x_idx


def squared_error_thresholds(
    alpha=1, r_star=5, slope_scale=5, N_neurons=39, R_t=245.41
):
    """XX = [alpha, r_star]?"""
    # bound
    if alpha <= 0.01 or alpha > 1:
        print(1e10)
        return 1e10
    # bound
    if r_star <= 0.01 or r_star > 50:
        print(1e10)
        return 1e10

    fit_ = value_efficient_coding_moment(
        N_neurons=N_neurons,
        R_t=R_t,
        X_OPT_ALPH=alpha,
        slope_scale=slope_scale,
    )
    fit_.replace_with_pseudo()

    res_max = []
    res_min = []
    for i in range(len(fit_.neurons_)):
        res_max.append(np.max(fit_.neurons_[i]))
        res_min.append(np.min(fit_.neurons_[i]))
    g_x_rstar = r_star

    # check if the g_x_rstar passes every data points
    check_ = []
    for i in range(len(fit_.neurons_)):
        check_.append(np.any(g_x_rstar <= fit_.neurons_[i]))

    if not np.all(check_):
        print(1e10)
        return 1e10  # just end here

    thresholds, _ = fit_.get_thresholds(
        r_star=g_x_rstar,
    )

    loss_1 = np.mean(
        (np.sort(np.array(thresholds)) - np.sort(np.array(fit_.Dabneys[1]))) ** 2
    )
    print(loss_1)
    return loss_1


def get_loss_r_star_slope(alpha=1):
    """pars=[r_star, slope]"""

    def loss(pars):
        return squared_error_thresholds(
            r_star=pars[0], slope_scale=pars[1], alpha=alpha
        )

    return loss


def fit_rstar_slope(alpha_dir="res_alpha", savedir="res_rstar_slope/"):
    # fitting thresholds by adjusting overall slope and r_star

    # load the distribution fit (find the alpha)
    LSE_s = []
    x_opt_s = []

    for ii in range(10):
        with open(alpha_dir + "res_fit{0}.pkl".format(ii), "rb") as f:
            data_1 = pkl.load(f)

        for i in range(len(data_1["res_s"])):
            LSE_s.append(data_1["res_s"][i]["fun"])
            x_opt_s.append(data_1["res_s"][i]["x"])

    id = np.argmin(LSE_s)
    alpha = x_opt_s[id]

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # make paramters using meshgrid
    num_seed = 5
    XX0 = np.linspace(0.02, 0.99, num_seed).tolist()
    XX1 = np.linspace(1.5, 15, num_seed).tolist()
    XX = np.array(np.meshgrid(XX0, XX1)).T.reshape(-1, 2)

    loss = get_loss_r_star_slope(alpha)

    for ij in range(num_seed**2):
        if not os.path.exists(savedir + "res_rstar_slope_{0}.pkl".format(ij)):
            # parameter seeds
            print("starting fit " + str(ij))
            print(XX[ij])
            t0 = time.time()
            res = minimize(
                loss,
                XX[ij],
                options={"maxiter": 1e5, "disp": True},
            )
            t1 = time.time()
            print("!!!!! {}s !!!!!".format(t1 - t0))

            with open(
                savedir + "res_rstar_slope_{0}.pkl".format(ij),
                "wb",
            ) as f:
                pkl.dump({"res": res}, f)


def load_matlab(filename):
    data = sio.loadmat(filename)
    return data


if __name__ == "__main__":
    fit_sigmoids()
    fit_alpha()
    fit_rstar_slope()
