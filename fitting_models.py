import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, uniform, gamma, lognorm
import os
import seaborn as sns
import time
import scipy.io as sio
import scipy

# efficient coding using sigmoid response functions


class value_efficient_coding_moment():
    def __init__(
            self,
            N_neurons=18,
            R_t=247.0690,
            X_OPT_ALPH=1.0,
            slope_scale=4,
            simpler=False):
        # real data prior
        self.offset = 0
        # it is borrowed from the original data of Dabney's
        self.juice_magnitudes = np.array(
            [.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array([
            0.06612594, 0.09090909, 0.14847358, 0.15489467,
            0.31159175, 0.1509519, 0.07705306])
        self.juice_prob /= np.sum(self.juice_prob)

        p_thresh = (2 * np.arange(N_neurons) + 1) / N_neurons / 2

        if simpler:  # to boost computation
            self.x = np.linspace(0, 30, num=int(1e3))
            self.x_inf = np.linspace(0, 300, num=int(1e4))
        else:
            self.x = np.linspace(0, 30, num=int(1e4))
            self.x_inf = np.linspace(0, 300, num=int(1e5))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        # np.linspace(-50, 50, num=int(1e4))
        self.x_log_inf = np.log(self.x_inf)

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 21]

        # logarithm space
        logmu = np.sum(np.log(self.juice_magnitudes) * self.juice_prob)
        logsd = np.sqrt(
            np.sum(((np.log(self.juice_magnitudes) - logmu) ** 2) * self.juice_prob))

        self.p_prior = lognorm.pdf(self.x, s=logsd, scale=np.exp(logmu))
        self.p_prior_inf = lognorm.pdf(
            self.x_inf, s=logsd, scale=np.exp(logmu))

        self.p_prior = lognorm.pdf(self.x, s=0.71, scale=np.exp(1.289))
        self.p_prior_inf = lognorm.pdf(self.x_inf, s=0.71, scale=np.exp(1.289))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / \
            np.sum(self.p_prior_inf * self._x_gap)

        # pseudo p-prior to make the sum of the p-prior in the range can be 1
        self.p_prior_pseudo = []
        ppp_cumsum = np.cumsum(self.p_prior_inf * self._x_gap)
        ppp_cumsum /= ppp_cumsum[-1]  # Offset
        self.p_prior_pseudo.append(ppp_cumsum[0])
        for i in range(len(ppp_cumsum) - 1):
            self.p_prior_pseudo.append(
                (ppp_cumsum[i + 1] - ppp_cumsum[i]) / self._x_gap)
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
        self.cum_P /= 1+1e-3

        # p_prior_inf_sum = self.p_prior_inf/np.sum(self.p_prior_inf)
        p_prior_inf_sum = self.p_prior_inf / np.sum(self.p_prior_inf)
        self.cum_P_pseudo = np.cumsum(
            p_prior_inf_sum)  # - 1e-5  # for approximation
        self.cum_P_pseudo /= 1+1e-3

        norm_d = self.p_prior / (1-self.cum_P)**(1-X_OPT_ALPH)
        NRMLZR = np.sum(norm_d * self._x_gap)
        norm_d = norm_d / NRMLZR

        cum_norm_D = np.cumsum(self.N * norm_d * self._x_gap)
        cum_norm_Dp = np.cumsum(self.N * norm_d * self._x_gap)/cum_norm_D[-1]

        thresh_ = np.interp(p_thresh, cum_norm_Dp, self.x)
        quant_ = np.interp(thresh_, self.x, cum_norm_Dp)

        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g = 1 / ((1 - self.cum_P)**X_OPT_ALPH)
        # norm_g /= NRMLZR
        norm_g /= self.N
        norm_g *= self.R

        norm_d_pseudo = self.p_prior_pseudo / \
            (1-self.cum_P_pseudo)**(1-X_OPT_ALPH)
        NRMLZR_pseudo = np.sum(norm_d_pseudo * self._x_gap)
        norm_d_pseudo = norm_d_pseudo / NRMLZR_pseudo

        cum_norm_D_pseudo = np.cumsum(self.N * norm_d_pseudo * self._x_gap)
        cum_norm_D_pseudop = np.cumsum(
            self.N * norm_d_pseudo * self._x_gap)/cum_norm_D_pseudo[-1]

        thresh_pseudo_ = np.interp(p_thresh, cum_norm_D_pseudop, self.x_inf)
        quant_pseudo_ = np.interp(
            thresh_pseudo_, self.x_inf, cum_norm_D_pseudop)

        norm_g_pseudo = 1 / \
            ((1 - self.cum_P_pseudo)**X_OPT_ALPH)
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
            self.neurons_.append(
                g_sn * scipy.special.betainc(a, b, cum_norm_Dp))
            self.gsn.append(g_sn)

            g_sn = norm_g_pseudo[np.argmin(
                np.abs(self.x_inf - self.sn_pseudo[i]))]

            a = slope_scale * quant_pseudo_[i]
            b = slope_scale * (1 - quant_pseudo_[i])
            self.neurons_pseudo_.append(
                g_sn * scipy.special.betainc(a, b, cum_norm_D_pseudop))
            self.gsn_pseudo.append(g_sn)

        # normalize afterward
        NRMLZR_G = self.R/np.sum(np.array(self.neurons_)
                                 * self.p_prior * self._x_gap)
        # neurons_arr=np.array(self.neurons_)*NRMLZR_G
        for i in range(len(self.neurons_)):
            self.neurons_[i] *= NRMLZR_G
            self.gsn[i] *= NRMLZR_G

        # normalize afterward
        NRMLZR_G_pseudo = self.R / \
            np.sum(np.array(self.neurons_pseudo_) *
                   self.p_prior_pseudo * self._x_gap)
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
        import scipy.io as sio
        fig5 = sio.loadmat("./measured_neurons/dabney_matlab/dabney_fit.mat")
        fig5_betas = sio.loadmat(
            "./measured_neurons/dabney_matlab/dabney_utility_fit.mat")
        zero_crossings = fig5['zeroCrossings_all'][:, 0]
        scaleFactNeg_all = fig5['scaleFactNeg_all'][:, 0]
        scaleFactPos_all = fig5['scaleFactPos_all'][:, 0]
        asymM_all = fig5['asymM_all'][:, 0]

        ZC_true_label = fig5['utilityAxis'].squeeze()

        def ZC_estimator(
            x): return fig5_betas["betas"][0, 0] + fig5_betas["betas"][1, 0] * x

        idx_to_maintain = np.where(
            (scaleFactNeg_all * scaleFactPos_all) > 0)[0]
        asymM_all = asymM_all[idx_to_maintain]
        idx_sorted = np.argsort(asymM_all)
        asymM_all = asymM_all[idx_sorted]
        estimated_ = np.array(self.get_quantiles_RPs(asymM_all))
        zero_crossings_ = fig5['zeroCrossings_all'][:, 0]
        zero_crossings_ = zero_crossings_[idx_to_maintain]
        zero_crossings_ = zero_crossings_[idx_sorted]
        zero_crossings_estimated = ZC_estimator(zero_crossings_)

        # ver 2
        dir_measured_neurons = 'measured_neurons/'
        NDAT = sio.loadmat(dir_measured_neurons + 'data_max.mat')['dat']
        zero_crossings_estimated = NDAT['ZC'][0, 0].squeeze()

        self.Dabneys = [estimated_, zero_crossings_estimated]

    def plot_approximate_kinky(self, r_star=0.02, name='gamma'):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        for i in range(self.N):  # excluded the last one since it is noisy
            (E_hn_prime_lower, E_hn_prime_higher, quantile), (x,
                                                              y), theta_x = self.hn_approximate(i, r_star)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))

        if name == 'gamma':
            plt.ylim((0, .10))
            plt.xlim((0.5, 2.5))
        elif name == 'normal':
            plt.ylim((0, .10))
            plt.xlim((3, 6))
        plt.title('Approximated')
        plt.show()
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(name + '/' + 'Approximated.png')
        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('quantile for each neuron')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(name + '/' + 'quantile for each neuron.png')

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

        return (E_hn_prime_lower, E_hn_prime_higher, E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower)), (
            self.x, np.array(out_)), theta_x

    def plot_approximate_kinky_normalized(self, r_star=0.02, name='gamma'):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        theta_s = []
        for i in range(self.N - 1):  # excluded the last one since it is noisy
            (E_hn_prime_lower, E_hn_prime_higher, quantile), (x,
                                                              y), theta_x = self.hn_approximate_normalized(i, r_star)
            theta_s.append(theta_x)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))

        if name == 'gamma':
            plt.ylim((0, .1))
            plt.xlim((0, 4))
        elif name == 'normal':
            plt.ylim((-.0, .20))
            plt.xlim((3, 6.5))
        plt.title('Approximated')
        plt.show()
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(name + '/' + 'Approximated normalized.png')
        plt.figure()
        xlocs = np.linspace(1, self.N - 1, self.N - 1)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('quantile for each neuron')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(name + '/' + 'expectile for each neuron normalized.png')

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
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap)
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / \
            np.sum(denominator_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i + 1] - hn[i]))
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap)
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap)
        E_hn_prime_higher = np.sum(
            inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (E_hn_prime_lower, E_hn_prime_higher, E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower)), (
            self.x, np.array(out_)), theta_x

    def plot_approximate_kinky_true(self, r_star=0.02, name='gamma'):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        theta_s = []
        alpha_s = []
        x_s = []
        y_s = []
        for i in range(self.N):  # excluded the last one since it is noisy
            (E_hn_prime_lower, E_hn_prime_higher, quantile), (x,
                                                              y), theta_x = self.hn_approximate_true(i, r_star)
            theta_s.append(theta_x)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))
            alpha_s.append([E_hn_prime_lower, E_hn_prime_higher])
            x_s.append(x)
            y_s.append(y)

        if name == 'gamma':
            plt.ylim((0.0, .15))
            plt.xlim((0, 8))
        elif name == 'normal':
            plt.ylim((0, .10))
            plt.xlim((2, 7))
        else:
            print('nothing')
        plt.title('Approximated')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'Approximated true.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('quantile for each neuron')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'expectile for each neuron true.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(self.sn_pseudo))
        for i, v in enumerate(np.array(self.sn_pseudo)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('preferred reward')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'preferred reward.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(theta_s))
        for i, v in enumerate(np.array(theta_s)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('reversal points')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'reversal points.png')

        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples_fitting_only(self, samples_idx, neurons, name, r_star_param=0.02):
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

        from scipy.optimize import curve_fit
        def func(rp, offset): return lambda x, a: a * (x - rp) + offset

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
            popt_minus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_minus[i])[idx_sort],
                                      np.array(R_minus[i])[idx_sort])
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_plus[i])[idx_sort],
                                     np.array(R_plus[i])[idx_sort])
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (np.divide(np.array(alpha_s), np.sum(
            np.array(alpha_s), 1).reshape(-1, 1)))[:, 1]
        x_s = []
        y_s = []
        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples_fitting_only_raw_rstar(self, samples_idx, neurons, name, r_star_param):
        # cal r-star
        # r_star = r_star_param
        # cal r-star
        # r_star = r_star_param

        from scipy.optimize import curve_fit
        def func(rp, offset): return lambda x, a: a * (x - rp) + offset

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
            popt_minus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_minus[i])[idx_sort],
                                      np.array(R_minus[i])[idx_sort])
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_plus[i])[idx_sort],
                                     np.array(R_plus[i])[idx_sort])
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
        quantiles = (np.divide(np.array(alpha_s), np.sum(
            np.array(alpha_s), 1).reshape(-1, 1)))[:, 1]
        x_s = []
        y_s = []
        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsim_fitting_only_raw_rstar(self, neurons, name, r_star_param, num_samples=int(1e4)):
        # cal r-star
        # r_star = r_star_param
        # cal r-star
        # r_star = r_star_param

        from scipy.optimize import curve_fit
        def func(rp, offset): return lambda x, a: a * (x - rp) + offset

        # get threshold for each neuron
        thresholds = []
        thresholds_idx = []
        for i in range(self.N):
            r_star = r_star_param[i]
            hn = neurons[i]
            ind = np.argmin(abs(hn - r_star))
            thresholds.append(self.x[ind])
            thresholds_idx.append(ind)

        alpha_plus = []
        alpha_minus = []
        alphas = []
        num_samples_yours = [num_samples]*self.N
        for i in range(self.N):
            # check it before
            while not np.all(int(num_samples_yours[i] * self.cum_P_pseudo[thresholds_idx[i]])):
                num_samples_yours[i] *= 10
                if num_samples_yours[i] > 1e6:
                    return False, 0, 0, 0, 0, 0

            while not np.all(int(num_samples_yours[i] * (1-self.cum_P_pseudo[thresholds_idx[i]]))):
                num_samples_yours[i] *= 10
                if num_samples_yours[i] > 1e6:
                    return False, 0, 0, 0, 0, 0

            sample_neg = []
            R_neg = []
            for s_id in range(int(num_samples_yours[i] * self.cum_P_pseudo[thresholds_idx[i]])):
                rand_neg = np.random.choice(np.arange(
                    0, thresholds_idx[i]), p=self.p_prior_inf[:thresholds_idx[i]] / np.sum(self.p_prior_inf[:thresholds_idx[i]]))
                sample_neg.append(self.x[rand_neg])
                R_neg.append(self.neurons_[i][rand_neg])
            sample_pos = []
            R_pos = []
            for s_id in range(int(num_samples_yours[i] * (1 - self.cum_P_pseudo[thresholds_idx[i]]))):
                rand_pos = np.random.choice(np.arange(thresholds_idx[i], len(self.x)),
                                            p=self.p_prior_inf[thresholds_idx[i]:] / np.sum(self.p_prior_inf[thresholds_idx[i]:]))
                sample_pos.append(self.x[rand_pos])
                R_pos.append(self.neurons_[i][rand_pos])
            neg_idx = np.argsort(sample_neg)
            popt_v1_minus, _ = curve_fit(func(self.x[thresholds_idx[i]], r_star), np.array(sample_neg)[neg_idx],
                                         np.array(R_neg)[neg_idx])
            pos_idx = np.argsort(sample_pos)
            popt_v1_plus, _ = curve_fit(func(self.x[thresholds_idx[i]], r_star), np.array(sample_pos)[pos_idx],
                                        np.array(R_pos)[pos_idx])

            alpha_minus.append(*popt_v1_minus)
            alpha_plus.append(*popt_v1_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (np.divide(np.array(alpha_s), np.sum(
            np.array(alpha_s), 1).reshape(-1, 1)))[:, 1]
        x_s = []
        y_s = []
        return True, quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples_fitting_only_rtimesg(self, samples_idx, neurons, name, gx,
                                                                r_star_param=0.02):
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

        from scipy.optimize import curve_fit
        def func(rp, offset): return lambda x, a: a * (x - rp) + offset

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
            popt_minus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_minus[i])[idx_sort],
                                      np.array(R_minus[i])[idx_sort])
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_plus[i])[idx_sort],
                                     np.array(R_plus[i])[idx_sort])
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (np.divide(np.array(alpha_s), np.sum(
            np.array(alpha_s), 1).reshape(-1, 1)))[:, 1]
        x_s = []
        y_s = []
        return quantiles, theta_s, alpha_s, x_s, y_s

    def plot_approximate_kinky_fromsamples(self, samples_idx, neurons, name, r_star_param=0.02):
        # cal r-star
        if r_star_param < 1:
            res_max = []
            res_min = []
            for i in range(len(self.neurons_)):
                res_max.append(np.max(self.neurons_[i]))
                res_min.append(np.min(self.neurons_[i]))
            r_star = np.min(res_max) * .8
        else:
            r_stars = np.copy(self.gsn_pseudo) / 2

        from scipy.optimize import curve_fit
        def func(rp, offset): return lambda x, a: a * (x - rp) + offset

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
            popt_minus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_minus[i])[idx_sort],
                                      np.array(R_minus[i])[idx_sort])
            alpha_minus.append(*popt_minus)
            idx_sort = np.argsort(X_plus[i])
            popt_plus, _ = curve_fit(func(thresholds[i], r_star), np.array(X_plus[i])[idx_sort],
                                     np.array(R_plus[i])[idx_sort])
            alpha_plus.append(*popt_plus)
            alphas.append([alpha_minus[-1], alpha_plus[-1]])

        theta_s = thresholds
        alpha_s = alphas
        quantiles = (np.divide(np.array(alpha_s), np.sum(
            np.array(alpha_s), 1).reshape(-1, 1)))[:, 1]
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
                    y_.append((self.x[xi] - thresholds[i])
                              * alpha_s[i][1] + r_star)
                else:
                    y_.append((self.x[xi] - thresholds[i])
                              * alpha_s[i][0] + r_star)

            plt.plot(x_, y_, color=str(colors[i]))
            x_s.append(x_)
            y_s.append(y_)

        plt.title('Approximated')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'Approximated true ' +
                    str(r_star_param) + ' .png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('quantile for each neuron')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' +
                    'expectile for each neuron true ' + str(r_star_param) + '.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(self.sn_pseudo))
        for i, v in enumerate(np.array(self.sn_pseudo)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('preferred reward')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'preferred reward ' +
                    str(r_star_param) + '.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(theta_s))
        for i, v in enumerate(np.array(theta_s)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('reversal points')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'reversal points ' +
                    str(r_star_param) + '.png')

        return quantiles, theta_s, alpha_s, x_s, y_s

    # def plot_approximate_variable_fromsamples_rstars(self, xy_to_plots, quantiles, theta_s, name='real_bigger'):
    #     plt.figure()
    #     colors = np.linspace(0, 0.7, self.N)
    #     for i in range(self.N):  # excluded the last one since it is noisy
    #         (x, y) = xy_to_plots[i]
    #         plt.plot(x, y, color=str(colors[i]))
    #
    #     plt.title('Approximated')
    #     if not os.path.exists(name + '/'):
    #         os.makedirs(name + '/')
    #     plt.savefig('./' + name + '/' + 'Approximated true.png')
    #
    #     plt.figure()
    #     xlocs = np.linspace(1, self.N, self.N)
    #     plt.bar(xlocs, np.array(quantiles))
    #     for i, v in enumerate(np.array(quantiles)):
    #         plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
    #     plt.title('quantile for each neuron')
    #     if not os.path.exists(name + '/'):
    #         os.makedirs(name + '/')
    #     plt.savefig('./' + name + '/' + 'expectile for each neuron true.png')
    #
    #     plt.figure()
    #     xlocs = np.linspace(1, self.N, self.N)
    #     plt.bar(xlocs, np.array(self.sn_pseudo))
    #     for i, v in enumerate(np.array(self.sn_pseudo)):
    #         plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
    #     plt.title('preferred reward')
    #     if not os.path.exists(name + '/'):
    #         os.makedirs(name + '/')
    #     plt.savefig('./' + name + '/' + 'preferred reward.png')
    #
    #     plt.figure()
    #     xlocs = np.linspace(1, self.N, self.N)
    #     plt.bar(xlocs, np.array(theta_s))
    #     for i, v in enumerate(np.array(theta_s)):
    #         plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
    #     plt.title('reversal points')
    #     if not os.path.exists(name + '/'):
    #         os.makedirs(name + '/')
    #     plt.savefig('./' + name + '/' + 'reversal points.png')
    #
    #     return quantiles, theta_s

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
                self.p_prior[i] * (hn[i]) * self._x_gap)  # this will be y value
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap)  # this will be x value
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / \
            np.sum(denominator_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap)
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap)  # this will be x value
        E_hn_prime_higher = np.sum(
            inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

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

        return (E_hn_prime_lower, E_hn_prime_higher, E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower)), (
            self.x, np.array(out_)), theta_x

    def plot_approximate_kinky_true_wo_prior(self, r_star=0.02, name='gamma'):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        theta_s = []
        for i in range(self.N):  # excluded the last one since it is noisy
            (E_hn_prime_lower, E_hn_prime_higher, quantile), (x, y), theta_x = self.hn_approximate_true_wo_prior(i,
                                                                                                                 r_star)
            theta_s.append(theta_x)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))

        if name == 'gamma':
            plt.ylim((0.01, .02))
            plt.xlim((0, 4))
        elif name == 'normal':
            plt.ylim((0, .10))
            plt.xlim((2, 7))
        plt.title('Approximated')
        plt.show()
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(name + '/' + 'Approximated true wo prior.png')
        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('quantile for each neuron')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(name + '/' + 'quantile for each neuron true wo prior.png')

        return quantiles

    def hn_approximate_true_wo_prior(self, n, r_star=0.02):
        hn = self.neurons_[n]
        theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap)  # this will be y value
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap)
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

        return (E_hn_prime_lower, E_hn_prime_higher, E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower)), (
            self.x, np.array(out_)), theta_x

    def plot_neurons(self, name='gamma'):
        # plot neurons response functions
        colors = np.linspace(0, 0.7, self.N)
        plt.figure()
        ymax = []
        for i in range(self.N):
            plt.plot(self.x, self.neurons_[i], color=str(colors[i]))
            ymax.append(self.neurons_[i][1499])
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title('Response functions of {0} neurons'.format(self.N))
        if not os.path.exists('./' + name + '/'):
            os.makedirs('./' + name + '/')
        plt.savefig('./' + name + '/' +
                    'Response functions of {0} neurons 300.png'.format(self.N))
        plt.xlim([0, 45])
        plt.ylim([0, np.max(ymax)])
        plt.savefig('./' + name + '/' +
                    'Response functions of {0} neurons.png'.format(self.N))

    def plot_neurons_pseudo(self, name='gamma'):
        # plot neurons response functions
        colors = np.linspace(0, 0.7, self.N)
        plt.figure()
        for i in range(self.N):
            plt.plot(self.x, self.neurons_pseudo_[i], color=str(colors[i]))
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title('Response functions of {0} neurons'.format(self.N))
        plt.show()
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig(
            name + '/' + 'Response functions of {0} neurons pseudo.png'.format(self.N))

    def plot_others(self, name='gamma'):
        ind30 = np.argmin(np.abs(self.x - 30))

        plt.figure()
        plt.title('Prior distribution')
        plt.plot(self.x, self.p_prior)
        if not os.path.exists('./' + name + '/'):
            os.makedirs('./' + name + '/')
        plt.savefig('./' + name + '/' + 'Prior distribution 300.png')
        plt.figure()
        plt.title('Density function')
        plt.plot(self.x, self.d_x_pseudo)
        plt.savefig('./' + name + '/' + 'Density function 300.png')
        plt.figure()
        plt.title('Gain function')
        plt.plot(self.x, self.g_x_pseudo)
        plt.savefig('./' + name + '/' + 'Gain function 300.png')

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

    # def
    def replace_with_pseudo(self):
        self.sn = self.sn_pseudo
        self.neurons_ = self.neurons_pseudo_
        self.x = self.x_inf
        self.p_prior = self.p_prior_inf
        # self.d_x = self.d_x_pseudo
        # self.g_x = self.g_x_pseudo

    def plot_approximate_variable_rstar(self, xy_to_plots, quantiles, theta_s, name='real_bigger'):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        for i in range(self.N):  # excluded the last one since it is noisy
            (x, y) = xy_to_plots[i]
            plt.plot(x, y, color=str(colors[i]))

        plt.title('Approximated')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'Approximated true.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('quantile for each neuron')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'expectile for each neuron true.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(self.sn_pseudo))
        for i, v in enumerate(np.array(self.sn_pseudo)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('preferred reward')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'preferred reward.png')

        plt.figure()
        xlocs = np.linspace(1, self.N, self.N)
        plt.bar(xlocs, np.array(theta_s))
        for i, v in enumerate(np.array(theta_s)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 2)))
        plt.title('reversal points')
        if not os.path.exists(name + '/'):
            os.makedirs(name + '/')
        plt.savefig('./' + name + '/' + 'reversal points.png')

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
                    self.p_prior[i] * (hn[i]) * self._x_gap)  # this will be y value
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior[i] * self.x[i] * self._x_gap)  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_lower = np.nan
            else:
                E_hn_prime_lower = np.sum(
                    inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

            # higher than theta
            inner_sigma_hn_prime = []
            denominator_hn_prime = []
            for i in range(theta, len(self.x) - 1):
                inner_sigma_hn_prime.append(
                    self.p_prior[i] * (hn[i]) * self._x_gap)
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior[i] * self.x[i] * self._x_gap)  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_higher = np.nan
            else:
                E_hn_prime_higher = np.sum(
                    inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

            # plot it
            out_ = []
            for i in range(len(self.x)):
                if i < theta:
                    out_.append(E_hn_prime_lower *
                                (self.x[i] - theta_x) + r_star)
                else:
                    out_.append(E_hn_prime_higher *
                                (self.x[i] - theta_x) + r_star)

            alpha_minus.append(E_hn_prime_lower)
            alpha_plus.append(E_hn_prime_higher)
            quantile.append(E_hn_prime_higher /
                            (E_hn_prime_higher + E_hn_prime_lower))
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
                    self.p_prior_inf[i] * (hn[i]) * self._x_gap)  # this will be y value
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior_inf[i] * self.x_inf[i] * self._x_gap)  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_lower = np.nan
            else:
                E_hn_prime_lower = np.sum(
                    inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

            # higher than theta
            inner_sigma_hn_prime = []
            denominator_hn_prime = []
            for i in range(theta, len(self.x_inf) - 1):
                inner_sigma_hn_prime.append(
                    self.p_prior_inf[i] * (hn[i]) * self._x_gap)
                # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
                denominator_hn_prime.append(
                    self.p_prior_inf[i] * self.x_inf[i] * self._x_gap)  # this will be x value
            if len(inner_sigma_hn_prime) == 0:
                E_hn_prime_higher = np.nan
            else:
                E_hn_prime_higher = np.sum(
                    inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

            # plot it
            out_ = []
            # for i in range(len(self.x)):
            #     if i < theta:
            #         out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            #     else:
            #         out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)
            for i in range(len(self.x_inf)):
                if i < theta:
                    out_.append(E_hn_prime_lower *
                                (self.x_inf[i] - theta_x) + r_star)
                else:
                    out_.append(E_hn_prime_higher *
                                (self.x_inf[i] - theta_x) + r_star)

            alpha_minus.append(E_hn_prime_lower)
            alpha_plus.append(E_hn_prime_higher)
            quantile.append(E_hn_prime_higher /
                            (E_hn_prime_higher + E_hn_prime_lower))
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
                self.p_prior[i] * (hn[i]) * self._x_gap)  # this will be y value
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap)  # this will be x value
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime) / \
            np.sum(denominator_hn_prime)

        # higher than theta
        inner_sigma_hn_prime = []
        denominator_hn_prime = []
        for i in range(theta, len(self.x) - 1):
            inner_sigma_hn_prime.append(
                self.p_prior[i] * (hn[i]) * self._x_gap)
            # denominator_hn_prime.append(self.p_prior[i] * self._x_gap) # this will be x value
            denominator_hn_prime.append(
                self.p_prior[i] * self.x[i] * self._x_gap)  # this will be x value
        E_hn_prime_higher = np.sum(
            inner_sigma_hn_prime) / np.sum(denominator_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i] - theta_x) + r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i] - theta_x) + r_star)

        return (E_hn_prime_lower, E_hn_prime_higher, E_hn_prime_higher / (E_hn_prime_higher + E_hn_prime_lower)), (
            self.x, np.array(out_)), theta_x

    def gen_simulation_data(self, num_samples=100):
        # sampling values
        sample_x = []
        sample_x_idx = []
        num_samples = num_samples
        for s in range(num_samples):
            sample_x.append(np.random.choice(
                self.x, p=self.p_prior / np.sum(self.p_prior)))
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
            sample_x.append(np.random.choice(
                self.x, p=self.p_prior / np.sum(self.p_prior)))
            sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])
            response_ = []
            for n_id in range(self.N):
                if np.isinf(num_samples_pp):
                    # get respones if you say infinite number of samples
                    response_.append(neuron[n_id][sample_x_idx[-1]])
                else:
                    pp_response = []
                    for pp_id in range(num_samples_pp):
                        mu = neuron[n_id][sample_x_idx[-1]]
                        x = np.arange(poisson.ppf(1e-5, mu),
                                      poisson.ppf(1 - 1e-5, mu))
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
            sample_x.append(np.random.choice(
                self.x, p=self.p_prior / np.sum(self.p_prior)))
            sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])

        self.sample_x = sample_x
        self.sample_x_idx = sample_x_idx
        return self.sample_x, self.sample_x_idx


class value_dist_rl(value_efficient_coding_moment):
    def __init__(self, ec, alphas, quantiles, thresholds, r_star, N_neurons=40, dir_save='drl'):
        self.dir_save = './' + dir_save + '/'

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        self.ec = ec

        self.N = N_neurons

        self.neurons_ = []
        self.neurons_mixed_ = []

        self.N_neurons = N_neurons
        self.r_star = r_star

        self.alphas = alphas
        # multiply something to make it done
        # self.alphas *= 100
        self.quantiles = quantiles
        self.thresholds = thresholds
        self.thresholds_mixed = thresholds

        self.offset = 0
        self.juice_magnitudes = np.array(
            [.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
                                    0.31159175, 0.1509519,
                                    0.07705306])  # it is borrowed from the original data of Dabney's
        self.juice_prob /= np.sum(self.juice_prob)
        self.x = np.linspace(0, 30, num=int(1e3))
        self.x_inf = np.linspace(0, 300, num=int(1e4))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        # np.linspace(-50, 50, num=int(1e4))
        self.x_log_inf = np.log(self.x_inf)

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 21]

        # sum_pdf = np.zeros(self.x.shape)
        sum_pdf = np.ones(self.x_log.shape) * 1e-8
        for i in range(len(self.juice_prob)):
            temp_ = norm.pdf(self.x_log, np.log(self.juice_magnitudes[i]), .2)
            sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]

        self.p_prior = sum_pdf / np.sum(sum_pdf * self._x_gap)

        # sum_pdf = np.zeros(self.x_inf.shape)
        sum_pdf = np.ones(self.x_log_inf.shape) * 1e-10
        for i in range(len(self.juice_prob)):
            temp_ = norm.pdf(self.x_log_inf, np.log(
                self.juice_magnitudes[i]), .2)
            sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]

        self.p_prior_inf = sum_pdf / np.sum(sum_pdf * self._x_gap)

        import pickle as pkl
        with open('lognormal_params.pkl', 'rb') as f:
            param = pkl.load(f)
        mu_mle = param['mu_mle']
        sigma_mle = param['sigma_mle']

        self.p_prior = lognorm.pdf(self.x, s=sigma_mle, scale=np.exp(mu_mle))
        self.p_prior_inf = lognorm.pdf(
            self.x_inf, s=sigma_mle, scale=np.exp(mu_mle))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / \
            np.sum(self.p_prior_inf * self._x_gap)

        self.p_prior_cdf = np.cumsum(self.p_prior)
        self.p_prior_cdf[-1] = 1 + 1e-16

        # self.x = np.linspace(0, 10, num=int(1e3))
        # self.x_inf = np.linspace(0, 1000, num=int(1e5))
        # self.p_prior = (gamma.pdf(self.x, 2))
        # self.p_prior_cdf = np.cumsum(self.p_prior)
        # self.p_prior_cdf[-1] = 1 + 1e-16
        # self.p_prior_inf = (gamma.pdf(self.x_inf, 2))
        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 1]

        for i in range(N_neurons):
            each_neuron = []
            for j in range(len(self.x)):
                # thresholds[i]
                if self.x[j] < thresholds[i]:
                    # slope * (x - threshold) + r_star
                    each_neuron.append(
                        alphas[i][0] * (self.x[j] - thresholds[i]) + r_star)
                else:
                    each_neuron.append(
                        alphas[i][1] * (self.x[j] - thresholds[i]) + r_star)
            # each_neuron_array = np.array(each_neuron)
            # each_neuron_array[np.where(each_neuron>=np.max(self.ec.neurons_[i]))] = np.max(self.ec.neurons_[i])
            # each_neuron = each_neuron_array.tolist()
            self.neurons_.append([each_neuron])
        np.random.seed(2021)
        index_ = np.linspace(0, N_neurons - 1, N_neurons).astype(np.int16)
        index = np.random.permutation(index_)

        self.alphas_mixed = []
        self.quantiles_mixed = []
        self.thresholds_mixed = []
        for i in range(N_neurons):
            self.neurons_mixed_.append(self.neurons_[index[i]])
            self.alphas_mixed.append(self.alphas[index[i]])
            self.quantiles_mixed.append(self.quantiles[index[i]])
            self.thresholds_mixed.append(self.thresholds[index[i]])
        pick_n_neurons = np.linspace(
            0, len(self.x) - 1, len(self.x)).astype(int)
        pick_n_neurons = np.random.permutation(pick_n_neurons)
        self.thresholds_mixed = self.x[pick_n_neurons[:self.N]]

    def init_figures(self, iternum):
        colors = np.linspace(0, 0.7, self.N)
        plt.figure(1)
        plt.close()
        plt.figure(1)
        for i in range(self.N):
            plt.plot(self.x, self.neurons_[i][0], color=str(colors[i]))
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title('Response functions of {0} neurons'.format(self.N))
        # plt.show()
        plt.savefig(self.dir_save +
                    'Response Function_{0:04d}.png'.format(iternum))

        plt.figure(2)
        plt.close()
        plt.figure(2)
        for i in range(self.N):
            plt.plot(self.x, self.neurons_mixed_[i][0], color=str(colors[i]))
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title('Response functions of mixed {0} neurons'.format(self.N))
        # plt.show()
        plt.savefig(self.dir_save +
                    'Mixed Response Function_{0:04d}.png'.format(iternum))

    def sample_value_cal_prediction_error(self, num_samples=100):
        # sampling values
        sample_x = []
        sample_x_idx = []
        num_samples = num_samples
        for s in range(num_samples):
            sample_x.append(np.random.choice(
                self.x, p=self.p_prior / np.sum(self.p_prior)))
            sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])
        self.sample_x = sample_x
        self.sample_x_idx = sample_x_idx

        # neurons
        E_PE = []
        PEs = []
        DAs = []
        for i in range(self.N_neurons):
            PE_each_neuron = []
            DA_each_neuron = []
            for j in range(num_samples):
                # y: self.neurons_[i][0][sample_x_idx[j]]
                PE_each_neuron.append(
                    self.x[sample_x_idx[j]] - self.thresholds[i])
                if self.x[sample_x_idx[j]] < self.thresholds[i]:  # negative prediction error
                    DA_each_neuron.append(
                        self.alphas[i][0] * (self.x[sample_x_idx[j]] - self.thresholds[i]))
                else:
                    DA_each_neuron.append(
                        self.alphas[i][1] * (self.x[sample_x_idx[j]] - self.thresholds[i]))
            PEs.append(PE_each_neuron)
            DAs.append(DA_each_neuron)
            E_PE.append(np.mean(PE_each_neuron))
        self.PEs = PEs
        self.DAs = DAs

        # neurons_mixed
        E_PE_mixed = []
        PEs_mixed = []
        DAs_mixed = []
        for i in range(self.N_neurons):
            PE_each_neuron = []
            DA_each_neuron = []
            for j in range(num_samples):
                # y: self.neurons_[i][0][sample_x_idx[j]]
                PE_each_neuron.append(
                    self.x[sample_x_idx[j]] - self.thresholds_mixed[i])
                if self.x[sample_x_idx[j]] < self.thresholds_mixed[i]:
                    DA_each_neuron.append(
                        self.alphas[i][0] * (self.x[sample_x_idx[j]] - self.thresholds_mixed[i]))
                else:
                    DA_each_neuron.append(
                        self.alphas[i][1] * (self.x[sample_x_idx[j]] - self.thresholds_mixed[i]))
            PEs_mixed.append(PE_each_neuron)
            DAs_mixed.append(DA_each_neuron)
            E_PE_mixed.append(np.mean(PE_each_neuron))
        self.PEs_mixed = PEs_mixed
        self.DAs_mixed = DAs_mixed

        return E_PE, E_PE_mixed

    def update_neurons(self):
        # update neurons
        self.neurons_ = []
        for i in range(self.N):
            each_neuron = []
            for j in range(len(self.x)):
                # thresholds[i]
                if self.x[j] < self.thresholds[i]:
                    # slope * (x - threshold) + r_star
                    each_neuron.append(
                        self.alphas[i][0] * (self.x[j] - self.thresholds[i]) + self.r_star)
                else:
                    each_neuron.append(
                        self.alphas[i][1] * (self.x[j] - self.thresholds[i]) + self.r_star)
                if np.any(np.isnan(np.array(each_neuron[-1]))):
                    print('a')
            # each_neuron_array = np.array(each_neuron)
            # each_neuron_array[np.where(each_neuron>=np.max(self.ec.neurons_[i]))] = np.max(self.ec.neurons_[i])
            # each_neuron = each_neuron_array.tolist()
            self.neurons_.append([each_neuron])

        # update neurons
        self.neurons_mixed_ = []
        for i in range(self.N):
            each_neuron = []
            for j in range(len(self.x)):
                # thresholds[i]
                if self.x[j] < self.thresholds_mixed[i]:
                    # slope * (x - threshold) + r_star
                    each_neuron.append(
                        self.alphas[i][0] * (self.x[j] - self.thresholds_mixed[i]) + self.r_star)
                else:
                    each_neuron.append(
                        self.alphas[i][1] * (self.x[j] - self.thresholds_mixed[i]) + self.r_star)
            # each_neuron_array = np.array(each_neuron)
            # each_neuron_array[np.where(each_neuron>=np.max(self.ec.neurons_[i]))] = np.max(self.ec.neurons_[i])
            # each_neuron = each_neuron_array.tolist()
            self.neurons_mixed_.append([each_neuron])

    def update_pes_using_sample(self):
        # For each i-th neuron update using expected DA
        self.thresholds_prev = np.copy(self.thresholds)
        self.thresholds_mixed_prev = np.copy(self.thresholds_mixed)

        # neurons
        for i in range(self.N_neurons):
            # self.thresholds[i] -= np.mean(self.DAs[i])
            self.thresholds[i] += np.mean(self.DAs[i])
            if self.thresholds[i] < 0:
                self.thresholds[i] = 0
        # neurons_mixed
        for i in range(self.N_neurons):
            # self.thresholds_mixed[i] -= np.mean(self.DAs_mixed[i])
            self.thresholds_mixed[i] += np.mean(self.DAs_mixed[i])
            if self.thresholds_mixed[i] < 0:
                self.thresholds_mixed[i] = 0

        # print('')

    def cal_cdfs_pdfs(self, iternum):
        # self.alphas # what contains the slopes.
        # self.thresholds# the value each neuron is coding.

        reversal_points = np.array(self.thresholds)
        alphas = np.array(self.alphas)
        idx = np.argsort(reversal_points)
        gotcdf = []
        gotcdf_x = []
        xi = 0
        for i in range(self.N_neurons):
            x = self.x[xi]
            while (x < reversal_points[idx[i]]):
                if (x > np.max(self.x)):
                    break
                if xi >= len(self.x) - 1:
                    break
                gotcdf.append(alphas[idx[i], 1] / np.sum(alphas[idx[i], :]))
                gotcdf_x.append(xi)
                xi += 1
                x = self.x[xi]
        gotcdf = np.array(gotcdf)
        gotcdf_x = np.array(gotcdf_x)
        plt.figure()
        plt.plot(gotcdf_x / 100, gotcdf)
        plt.xlim((0, 10))
        # plt.xticks(np.arange(0,10.1,step=2), ['0','2','4','6','8','10'])
        plt.xticks(np.arange(0, 10.1, step=2))
        plt.ylim((0, 1))
        plt.savefig(self.dir_save + 'CDF_{0:04d}.png'.format(iternum))

        plt.figure()
        gotpdf = np.ediff1d(gotcdf, to_begin=gotcdf[0])
        plt.plot(gotcdf_x / 100, gotpdf)
        plt.xlim((0, 10))
        plt.xticks(np.arange(0, 10.1, step=2))
        plt.ylim((0, 1))
        plt.savefig(self.dir_save + 'PDF_{0:04d}.png'.format(iternum))

        reversal_points_mixed = np.array(self.thresholds_mixed)
        alphas_mixed = np.array(self.alphas)
        idx = np.argsort(reversal_points_mixed)
        gotcdf_mixed = []
        gotcdf_mixed_x = []
        xi = 0
        for i in range(self.N_neurons):
            x = self.x[xi]
            while (x < reversal_points_mixed[idx[i]]):
                if (x > np.max(self.x)):
                    break
                if xi >= len(self.x) - 1:
                    break
                gotcdf_mixed.append(
                    alphas_mixed[idx[i], 1] / np.sum(alphas_mixed[idx[i], :]))
                gotcdf_mixed_x.append(xi)
                xi += 1
                x = self.x[xi]
        gotcdf_mixed = np.array(gotcdf_mixed)
        gotcdf_mixed_x = np.array(gotcdf_mixed_x)
        plt.figure()
        plt.plot(gotcdf_mixed_x / 100, gotcdf_mixed)
        plt.xlim((0, 10))
        # plt.xticks(np.arange(0,10.1,step=2), ['0','2','4','6','8','10'])
        plt.xticks(np.arange(0, 10.1, step=2))
        plt.ylim((0, 1))
        # plt.show()
        plt.savefig(self.dir_save + 'MIXED CDF_{0:04d}.png'.format(iternum))

        plt.figure()
        gotpdf_mixed = np.ediff1d(gotcdf_mixed, to_begin=gotcdf_mixed[0])
        plt.plot(gotcdf_mixed_x / 100, gotpdf_mixed)
        plt.xlim((0, 10))
        plt.xticks(np.arange(0, 10.1, step=2))
        # plt.ylim((0,1))
        plt.savefig(self.dir_save + 'MIXED PDF_{0:04d}.png'.format(iternum))

        # erasing wrong interpretations? later

    def plot_thresholds(self, iternum):
        plt.figure(998)
        plt.close()
        plt.figure(998)
        sns.set_style('whitegrid')
        sns.kdeplot(self.thresholds, bw=.75, color='k', lw=3., shade=True)
        sns.rugplot(self.thresholds, color='k')
        plt.xlim((0, 10))
        plt.ylim((0, .4))
        box_prop = dict(facecolor="wheat", alpha=1)
        plt.text(6, 0.1, 'mean:' + str(np.mean(self.thresholds)), bbox=box_prop)
        # plt.savefig('./drl/' + 'Value_{0:04d}.png'.format(iternum))
        plt.savefig(self.dir_save + 'RPs_{0:04d}.png'.format(iternum))

        plt.figure(999)
        plt.close()
        plt.figure(999)
        sns.set_style('whitegrid')
        sns.kdeplot(self.thresholds_mixed, bw=.75,
                    color='k', lw=3., shade=True)
        sns.rugplot(self.thresholds_mixed, color='k')
        plt.xlim((0, 10))
        plt.ylim((0, .4))
        box_prop = dict(facecolor="wheat", alpha=1)
        plt.text(6, 0.1, 'mean:' +
                 str(np.mean(self.thresholds_mixed)), bbox=box_prop)
        # plt.savefig('./drl/' + 'Mixed Value_{0:04d}.png'.format(iternum))
        plt.savefig(self.dir_save + 'Mixed RPs_{0:04d}.png'.format(iternum))

    def TODO(self):
        return 0


def get_If(neurons_, x):
    If_ = []
    x_gap = x[1] - x[0]
    for i in range(1, len(x)):
        inner = []
        for n in range(len(neurons_)):
            hn = neurons_[n]
            if len(hn) == 1:
                hn = hn[0]
            # if not np.any(np.isnan(hn)):
            if hn[i] > 0:
                inner.append((((hn[i] - hn[i - 1]) / x_gap) ** 2) / (hn[i]))
        If_.append(np.sum(inner))
        if np.sum(inner) > 20:
            print('why')
    return x[1:], If_


def neuron_neuron_offset(ec, neuron, start_offset, maxval=118.0502):
    flag = 0
    OFFSET = start_offset
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1
    flag = 0
    OFFSET = OFFSET - 2

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-1

    flag = 0
    OFFSET = OFFSET - 2e-1

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-2

    flag = 0
    OFFSET = OFFSET - 2e-2

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-4

    flag = 0
    OFFSET = OFFSET - 2e-4

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-5

    neuneurons_ = []
    for i in range(len(neuron)):
        neuneurons_.append(
            (neuron[i, :] + OFFSET).tolist())
    neuneurons_ = np.array(neuneurons_)

    return OFFSET, neuneurons_


class value_efficient_coding_fitting_sd(value_efficient_coding_moment):
    def __init__(self, prior='normal', N_neurons=18, R_t=247.0690, XX2=1.0):
        # real data prior
        self.offset = 0
        self.juice_magnitudes = np.array(
            [.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
                                    0.31159175, 0.1509519,
                                    0.07705306])  # it is borrowed from the original data of Dabney's
        self.juice_prob /= np.sum(self.juice_prob)

        self.x = np.linspace(0, 30, num=int(1e3))
        self.x_inf = np.linspace(0, 300, num=int(1e4))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        # np.linspace(-50, 50, num=int(1e4))
        self.x_log_inf = np.log(self.x_inf)

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 21]

        # logarithm space
        logmu = np.sum(np.log(self.juice_magnitudes) * self.juice_prob)
        logsd = np.sqrt(
            np.sum(((np.log(self.juice_magnitudes) - logmu) ** 2) * self.juice_prob))

        self.p_prior = lognorm.pdf(self.x, s=logsd, scale=np.exp(logmu))
        self.p_prior_inf = lognorm.pdf(
            self.x_inf, s=logsd, scale=np.exp(logmu))

        import pickle as pkl
        with open('lognormal_params.pkl', 'rb') as f:
            param = pkl.load(f)
        mu_mle = param['mu_mle']
        sigma_mle = param['sigma_mle']

        # self.p_prior = lognorm.pdf(self.x,s = sigma_mle,scale= np.exp(mu_mle))
        # self.p_prior_inf = lognorm.pdf(self.x_inf,s = sigma_mle,scale= np.exp(mu_mle))

        # with open('empirical_lognormal.pkl', 'rb') as f:
        #     param_emp = pkl.load(f)['param']
        # self.p_prior = lognorm.pdf(self.x,s = param_emp[0], loc = param_emp[1],scale = param_emp[2])
        # self.p_prior_inf = lognorm.pdf(self.x_inf,s = param_emp[0], loc = param_emp[1],scale = param_emp[2])

        self.p_prior = lognorm.pdf(self.x, s=0.71, scale=np.exp(1.289))
        self.p_prior_inf = lognorm.pdf(self.x_inf, s=0.71, scale=np.exp(1.289))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / \
            np.sum(self.p_prior_inf * self._x_gap)

        # p_prior_in_log = norm.pdf(self.x_log,log_mu,log_sd)
        # self.p_prior = (p_prior_in_log) / np.sum(p_prior_in_log*self._x_gap)

        # p_prior_in_log = norm.pdf(self.x_log_inf,log_mu,log_sd)
        # self.p_prior_inf = p_prior_in_log / np.sum(p_prior_in_log*self._x_gap)
        # self.p_prior = self.p_prior_inf[:1000]

        # pseudo p-prior to make the sum of the p-prior in the range can be 1
        self.p_prior_pseudo = []
        ppp_cumsum = np.cumsum(self.p_prior_inf * self._x_gap)
        ppp_cumsum /= ppp_cumsum[-1]  # Offset
        self.p_prior_pseudo.append(ppp_cumsum[0])
        for i in range(len(ppp_cumsum) - 1):
            self.p_prior_pseudo.append(
                (ppp_cumsum[i + 1] - ppp_cumsum[i]) / self._x_gap)
        self.p_prior_pseudo = np.array(self.p_prior_pseudo)

        # since we posit a distribution ranged in [0,20] (mostly) we hypothesized that integral from -inf to +inf is same
        # as the integral from 0 to 20 in this toy example. From now on, we just calculated cumulative distribution using
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
        self.cum_P = np.cumsum(p_prior_sum) - 1e-3  # for approximation
        # p_prior_inf_sum = self.p_prior_inf/np.sum(self.p_prior_inf)
        p_prior_inf_sum = self.p_prior_inf / np.sum(self.p_prior_inf)
        self.cum_P_pseudo = np.cumsum(
            p_prior_inf_sum) - 1e-5  # for approximation

        norm_d = self.p_prior ** XX2 / (1 - self.cum_P) ** (1 - XX2)
        NRMLZR = np.sum(norm_d * self._x_gap)
        norm_d = norm_d / NRMLZR

        cum_norm_D = np.cumsum(self.N * norm_d * self._x_gap)

        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g = self.p_prior ** (1 - XX2) / ((1 - self.cum_P) ** XX2)
        # norm_g /= NRMLZR
        norm_g /= self.N
        norm_g *= self.R

        norm_d_pseudo = self.p_prior_pseudo ** XX2 / \
            (1 - self.cum_P_pseudo) ** (1 - XX2)
        NRMLZR_pseudo = np.sum(norm_d_pseudo * self._x_gap)
        norm_d_pseudo = norm_d_pseudo / NRMLZR_pseudo

        cum_norm_D_pseudo = np.cumsum(self.N * norm_d_pseudo * self._x_gap)

        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g_pseudo = self.p_prior_pseudo ** (1 - XX2) / \
            ((1 - self.cum_P_pseudo) ** XX2)
        # norm_g /= NRMLZR
        norm_g_pseudo /= self.N
        norm_g_pseudo *= self.R
        #
        # # density & gain
        # self.d = lambda s: self.N * self.p_prior[s]
        # self.d_pseudo = lambda s: self.N * self.p_prior_pseudo[s]
        # self.g = lambda s: self.R / ((self.N) * (1 - self.cum_P[s]))
        # self.g_pseudo = lambda s: self.R / ((self.N) * (1 - self.cum_P_pseudo[s]))
        #
        # self.d_x = np.empty((0,))
        # self.d_x_pseudo = np.empty((0,))
        # self.g_x = np.empty((0,))
        # self.g_x_pseudo = np.empty((0,))
        # for j in range(len(self.x)):
        #     self.d_x = np.concatenate((self.d_x, np.array([self.d(j)])))
        #     self.g_x = np.concatenate((self.g_x, np.array([self.g(j)])))
        #     # self.d_x.append(self.d(i)) # based on the assumption that our domain ranged [0,20] is approximately same as [-inf, inf]
        #
        # for j in range(len(self.x_inf)):
        #     self.d_x_pseudo = np.concatenate((self.d_x_pseudo, np.array([self.d_pseudo(j)])))
        #     self.g_x_pseudo = np.concatenate((self.g_x_pseudo, np.array([self.g_pseudo(j)])))

        # find each neuron's location
        # preferred response of each neuron. It is x=0 in the prototype sigmoid function (where y=0.5)
        self.sn = []
        self.sn_pseudo = []

        self.D_pseudo = []

        # self.D = np.cumsum(self.d_x * self._x_gap)  # multiply _x_gap to approximate continuous integral.
        # self.D_pseudo = np.cumsum(self.d_x_pseudo * self._x_gap)  # pseudo version of it
        # # offset
        # self.D_pseudo += 0

        ind_sets = []
        ind_sets_pseudo = []
        for i in range(self.N):
            ind_set = np.argmin(np.abs(np.round(cum_norm_D + .5) - (i + 1)))
            # ind_set = np.argmin(np.abs((self.D+.1) - s(i + 1)))
            self.sn.append(self.x[np.min(ind_set)])  # take the minimum of two

            ind_sets.append(np.min(ind_set))

            # ind_set = np.argmin(np.abs(np.round(self.D_pseudo+.1) - (i + 1)))
            ind_set = np.argmin(np.abs((cum_norm_D_pseudo + .5) - (i + 1)))
            # i_isclose0 = np.squeeze(np.where(np.isclose((self.D_pseudo - (i+1)),0)))
            # ind_set = [i_argmin, *i_isclose0.tolist()]
            # take the minimum of two
            self.sn_pseudo.append(self.x_inf[np.min(ind_set)])

            ind_sets_pseudo.append(np.min(ind_set))

        # each neurons response function
        self.neurons_ = []  # self.N number of neurons

        # from e.q. (4.2) in ganguli et al. (2014)
        # first derivative of prototype sigmoid function
        def h_prime(s): return np.exp(-s) / ((1 + np.exp(-s)) ** 2)

        g_sns = []
        x_gsns = []
        self.gsn = []

        # for j in range(len(self.x)):
        #     # hn_inner_integral.append(self.d_x[j]*h_prime(self.D[j]-(i+1))*self._x_gap)
        #     print(self.D[j] - (i))
        locs = []
        hn_primes = []
        for i in range(self.N):

            locs.append(np.squeeze(np.where(self.x == self.sn[i])))
            g_sn = norm_g[np.squeeze(np.where(self.x == self.sn[i]))]
            hn_inner_integral = []
            for j in range(len(self.x)):
                # hn_inner_integral.append(self.d_x[j]*h_prime(self.D[j]-(i+1))*self._x_gap)
                # hn_inner_integral.append(self.d_x[j] * h_prime(self.D[j] - (i + 1)) * self._x_gap)
                hn_inner_integral.append(
                    norm_d[j] * h_prime(cum_norm_D[j] - (i + 1)) * self._x_gap)
            h_n = g_sn * np.cumsum(hn_inner_integral)
            self.neurons_.append(h_n)
            g_sns.append(g_sn)
            x_gsns.append(self.sn[i])
            self.gsn.append(g_sn)
            # hn_primes.append(h_prime(self.D[j] - (i + 1)))
            hn_primes.append(h_prime(cum_norm_D[j] - (i + 1)))

        g_sns = []
        x_gsns = []
        self.neurons_pseudo_ = []  # pseudo
        self.gsn_pseudo = []
        for i in range(self.N):
            # g_sn = self.g(np.squeeze(np.where(self.x == self.sn[i])))
            g_sn = norm_g_pseudo[np.squeeze(
                np.where(self.x_inf == self.sn_pseudo[i]))]
            hn_inner_integral = []
            for j in range(len(self.x_inf)):
                # hn_inner_integral.append(self.d_x_pseudo[j] * h_prime(self.D_pseudo[j] - (i + 1)) * self._x_gap)
                hn_inner_integral.append(
                    norm_d_pseudo[j] * h_prime(cum_norm_D_pseudo[j] - (i + 1)) * self._x_gap)
            h_n = g_sn * np.cumsum(hn_inner_integral)
            self.neurons_pseudo_.append(h_n)
            g_sns.append(g_sn)
            x_gsns.append(self.sn_pseudo[i])
            self.gsn_pseudo.append(g_sn)


class fitting_model_model():
    def __init__(self, dir_save_figures, samples_idx, fit, X_OPT_ALPH):
        self.dir_save_figures = dir_save_figures
        self.samples_idx = samples_idx
        self.fit_ = fit
        self.Dabneys = fit.Dabneys
        self.X_OPT_ALPH = X_OPT_ALPH

    # def import_ec_components(self, ec):
    #     self.ec = ec

    def get_quantiles_RPs(self, fit_, quantiles):
        P_PRIOR = np.cumsum(fit_.p_prior_inf * fit_._x_gap)
        RPs = []
        for i in range(len(quantiles)):
            indx = np.argmin(abs(P_PRIOR - quantiles[i]))
            RPs.append(fit_.x_inf[indx])
        return RPs

    def get_quantiles_RPs_fromDiscrete(self, fit_, quantiles):
        P_PRIOR = np.cumsum(fit_.p_prior_inf_dirac * fit_._x_gap)
        RPs = []
        for i in range(len(quantiles)):
            indx = np.argmin(abs(P_PRIOR - quantiles[i]))
            RPs.append(fit_.x_inf[indx])
        return RPs

    def neuron_R_fit_timesG_fixedR_paramLog(self, XX):
        # bound
        if XX[0] <= 0.01:
            XX[0] = 0.01
        if XX[0] > 1:
            XX[0] = 1
        # bound
        if XX[1] <= 0.01:
            XX[1] = 0.01
        if XX[1] > 50:
            XX[1] = 50

        print(XX)
        fit_ = value_efficient_coding_moment('./', N_neurons=self.fit_.N, R_t=self.fit_.R, X_OPT_ALPH=XX[0],
                                             slope_scale=5.07)
        fit_.replace_with_pseudo()

        res_max = []
        res_min = []
        for i in range(len(fit_.neurons_)):
            res_max.append(np.max(fit_.neurons_[i]))
            res_min.append(np.min(fit_.neurons_[i]))
        r_star = np.min(res_max)
        g_x_rstar = []
        for i in range(len(fit_.neurons_)):
            g_x_rstar.append(XX[1])

        # check if the g_x_rstar passes every data points
        check_ = []
        r_star_idx = []
        for i in range(len(fit_.neurons_)):
            check_.append(np.any(g_x_rstar[i] <= fit_.neurons_[i]))

        num_samples = int(1e3)
        # # check if there is any data that falls into the invalid range defined by g_x_rstar
        check_2 = []
        r_star_idx = []
        check_3 = []
        for i in range(len(fit_.neurons_)):
            temp_threshold = np.argmin(np.abs(fit_.neurons_[i] - g_x_rstar[i]))
            check_2.append(
                np.all([len(fit_.x[:temp_threshold]) > 1, len(fit_.x[temp_threshold:]) > 1]))
            check_3.append(
                int(num_samples * fit_.cum_P_pseudo[temp_threshold]))

        from scipy.optimize import curve_fit
        def func(rp, offset): return lambda x, a: a * (x - rp) + offset

        if not np.all(check_):
            return 1e10  # just end here

        if not np.all(check_2):
            return 1e10  # just end here

        tof, quantiles_constant, thresholds_constant, alphas, xs, ys = fit_.plot_approximate_kinky_fromsim_fitting_only_raw_rstar(
            fit_.neurons_,
            self.dir_save_figures, r_star_param=g_x_rstar, num_samples=num_samples)

        if tof:
            RPs = self.get_quantiles_RPs(fit_, quantiles_constant)
            loss_rp = np.log(
                np.mean(np.sum((np.array(thresholds_constant) - np.array(RPs)) ** 2)))
            loss_1 = np.log(
                np.mean(np.sum((np.array(thresholds_constant)-np.array(self.Dabneys[1]))**2)))
            print(loss_1)
            return loss_1
        else:
            print(1e10)
            return 1e10

    def neuron_R_fit_fixed_rstar(self, XX):
        fit_ = value_efficient_coding(N_neurons=39, R_t=XX)
        fit_.replace_with_pseudo()
        quantiles_constant, thresholds_constant, alphas, xs, ys = fit_.plot_approximate_kinky_fromsamples_fitting_only(
            self.samples_idx, fit_.neurons_,
            name=self.dir_save_figures, r_star_param=.8)
        RPs = self.get_quantiles_RPs(fit_, quantiles_constant)

        return np.mean((np.array(thresholds_constant) - np.array(RPs)) ** 2)


def neuron_neuron_scale(ec, neuron, start_offset, maxval=118.0502):
    flag = 0
    OFFSET = start_offset
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] * OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        if flag >= maxval:
            break
        OFFSET += 1
    flag = 0
    OFFSET = OFFSET - 1

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] * OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        if flag >= maxval:
            break
        OFFSET += 1e-1

    flag = 0
    OFFSET = OFFSET - 1e-1
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] * OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        if flag >= maxval:
            break
        OFFSET += 1e-2

    flag = 0
    OFFSET = OFFSET - 1e-2
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] * OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        if flag >= maxval:
            break
        OFFSET += 1e-4

    flag = 0
    OFFSET = OFFSET - 1e-4
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] * OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        if flag >= maxval:
            break
        OFFSET += 1e-5

    neuneurons_ = []
    for i in range(len(neuron)):
        neuneurons_.append(
            (neuron[i, :] + OFFSET).tolist())
    neuneurons_ = np.array(neuneurons_)

    return OFFSET, neuneurons_


def main():
    # fitting efficient code
    savedir = 'res_fit_to_empirical2/'

    # load the distribution fit (find the alpha)
    LSE = 99999
    LSE_s = []
    x_opt_s = []

    import pickle as pkl

    for ii in range(10):
        with open(savedir + 'res_fit{0}.pkl'.format(ii), 'rb') as f:
            data_1 = pkl.load(f)

        for i in range(len(data_1['res_s'])):
            LSE_s.append(data_1['res_s'][i]['fun'])
            x_opt_s.append(data_1['res_s'][i]['x'])
            if LSE > data_1['res_s'][i]['fun']:
                LSE = data_1['res_s'][i]['fun']
                x_opt = data_1['res_s'][i]['x']

    idxxx = np.argsort(LSE_s)
    id = np.argmin(LSE_s)

    N_neurons = 39
    R_t = 245.41
    dir_save_figures = './'
    print('Initiailze efficient coding part')
    ec = value_efficient_coding_moment(
        dir_save_figures, N_neurons=N_neurons, R_t=R_t, X_OPT_ALPH=x_opt_s[id], slope_scale=5.07)
    ec.replace_with_pseudo()

    # curve fitting part
    fit_class = fitting_model_model(dir_save_figures, [], ec, x_opt_s[id])
    from scipy.optimize import minimize, least_squares
    from scipy import optimize
    import time

    num_seed = 10
    XX0 = np.linspace(0, 1, num_seed).tolist()
    XX1 = np.linspace(1, 10, num_seed).tolist()
    from itertools import product

    import sys

    savedir = 'res_fit_alpha_fit/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    max_total = 1000
    # ii = int(sys.argv[1])
    ii = 0
    # if True:
    if not os.path.exists(savedir + 'res_fit_apprx_freealpha_freebeta_lognormal{0}.pkl'.format(ii)):
        if True:

            # make combination of XX0 and XX1

            # make paramters using meshgrid
            XX = np.array(np.meshgrid(XX0, XX1)).T.reshape(-1, 2)

            # parameter seeds
            print(ii)
            t0 = time.time()
            res_s = []
            for ii in range(len(XX)):
                XX_ = XX[ii]
                res = minimize(fit_class.neuron_R_fit_timesG_fixedR_paramLog, XX_, options={
                               'maxiter': 1e5, 'disp': True})
                res_s.append(res)
                t1 = time.time()
                print('!!!!! {}s !!!!!'.format(t1 - t0))

                import pickle as pkl
                with open(savedir + 'res_fit_apprx_freealpha_freebeta_lognormal{0}.pkl'.format(ii),
                          'wb') as f:
                    pkl.dump({'res_s': res_s}, f)


def load_matlab(filename):
    data = sio.loadmat(filename)
    return data


def neuron_neuron_offset(ec, neuron, start_offset, maxval=118.0502):
    flag = 0
    OFFSET = start_offset
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1
    flag = 0
    OFFSET = OFFSET - 2
    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-2

    flag = 0
    OFFSET = OFFSET - 2e-2

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-4

    flag = 0
    OFFSET = OFFSET - 2e-4

    while flag < maxval:
        neuneurons_ = []
        for i in range(len(neuron)):
            neuneurons_.append(
                (neuron[i, :] + OFFSET).tolist())
        neuneurons_ = np.array(neuneurons_)

        dx = ec.x_inf[1] - ec.x_inf[0]
        R_estimated = []
        for xi in range(len(neuneurons_[0])):
            R_estimated.append(
                ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
        flag = np.sum(R_estimated)
        print(flag)
        print(OFFSET)
        OFFSET += 1e-5

    neuneurons_ = []
    for i in range(len(neuron)):
        neuneurons_.append(
            (neuron[i, :] + OFFSET).tolist())
    neuneurons_ = np.array(neuneurons_)

    return OFFSET, neuneurons_


class statistics_():
    def __init__(self, N_neurons, R_t=247.0690):
        self.N = N_neurons

        self.neurons_ = []
        self.neurons_mixed_ = []

        self.N_neurons = self.N

        self.offset = 5
        self.juice_magnitudes = np.array(
            [.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
                                    0.31159175, 0.1509519,
                                    0.07705306])  # it is borrowed from the original data of Dabney's
        self.juice_prob /= np.sum(self.juice_prob)
        self.x = np.linspace(0, 30, num=int(750))
        self.x_inf = np.linspace(0, 300, num=int(7500))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        # np.linspace(-50, 50, num=int(1e4))
        self.x_log_inf = np.log(self.x_inf)

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 21]

        # sum_pdf = np.zeros(self.x.shape)
        sum_pdf = np.ones(self.x_log.shape) * 1e-8
        for i in range(len(self.juice_prob)):
            temp_ = norm.pdf(self.x_log, np.log(self.juice_magnitudes[i]), .2)
            sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]

        self.p_prior = sum_pdf / np.sum(sum_pdf * self._x_gap)

        # sum_pdf = np.zeros(self.x_inf.shape)
        sum_pdf = np.ones(self.x_log_inf.shape) * 1e-10
        for i in range(len(self.juice_prob)):
            temp_ = norm.pdf(self.x_log_inf, np.log(
                self.juice_magnitudes[i]), .2)
            sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]

        self.p_prior_inf = sum_pdf / np.sum(sum_pdf * self._x_gap)

        import pickle as pkl
        with open('lognormal_params.pkl', 'rb') as f:
            param = pkl.load(f)
        mu_mle = param['mu_mle']
        sigma_mle = param['sigma_mle']

        self.p_prior = lognorm.pdf(self.x, s=sigma_mle, scale=np.exp(mu_mle))
        self.p_prior_inf = lognorm.pdf(
            self.x_inf, s=sigma_mle, scale=np.exp(mu_mle))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / \
            np.sum(self.p_prior_inf * self._x_gap)

        # sum_pdf = np.zeros(self.x_inf.shape)
        sum_pdf = np.ones(self.x_inf.shape) * 1e-10
        for i in range(len(self.juice_prob)):
            temp_ = norm.pdf(self.x_inf, self.juice_magnitudes[i], 1e-3)
            sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]

        self.p_prior_inf_dirac = sum_pdf / np.sum(sum_pdf * self._x_gap)

        self.p_prior_cdf = np.cumsum(self.p_prior)
        self.p_prior_cdf[-1] = 1 + 1e-16

        self.R = R_t

    def get_quantiles_RPs(self, quantiles):
        P_PRIOR = np.cumsum(self.p_prior_inf * self._x_gap)
        RPs = []
        for i in range(len(quantiles)):
            indx = np.argmin(abs(P_PRIOR - quantiles[i]))
            RPs.append(self.x_inf[indx])
        return RPs

    def get_quantiles_RPs_fromDiscrete(self, quantiles):
        P_PRIOR = np.cumsum(self.p_prior_inf_dirac * self._x_gap)
        RPs = []
        for i in range(len(quantiles)):
            indx = np.argmin(abs(P_PRIOR - quantiles[i]))
            RPs.append(self.x_inf[indx])
        return RPs


def test():
    savedir = './res_fit_alpha_fit/'

    LSE = 99999
    LSE_s = []
    x_opt_s = []

    for ii in range(1000):
        try:
            with open(savedir + 'res_fit_apprx_freealpha_freebeta_lognormal{0}.pkl'.format(ii), 'rb') as f:
                data_1 = pkl.load(f)

            for i in range(len(data_1['res_s'])):
                LSE_s.append(data_1['res_s'][i]['fun'])
                x_opt_s.append(data_1['res_s'][i]['x'])
                if LSE > data_1['res_s'][i]['fun']:
                    LSE = data_1['res_s'][i]['fun']
                    x_opt = data_1['res_s'][i]['x']
        except:
            ''

    xss = np.array(x_opt_s)
    id = np.argmin(LSE_s)

    print(LSE_s[id])
    print(xss[id])

    ec_moment = value_efficient_coding_moment(
        './', N_neurons=39, R_t=245.41, X_OPT_ALPH=0.7665, slope_scale=5.07)

    ec_moment.replace_with_pseudo()
    g_x_rstar = []
    for i in range(len(ec_moment.neurons_)):
        g_x_rstar.append(xss[id][0])

    tf, quantiles_constant, thresholds_constant, alphas, xs, ys = ec_moment.plot_approximate_kinky_fromsim_fitting_only_raw_rstar(
        ec_moment.neurons_, '.', r_star_param=g_x_rstar, num_samples=int(1e4))

    print('thresholds')

    np.random.seed(2021)
    num_samples = 10
    samples, samples_idx = ec_moment.gen_samples(num_samples=int(num_samples))

    import scipy.io as sio
    fig5 = sio.loadmat("./measured_neurons/dabney_matlab/dabney_fit.mat")
    fig5_betas = sio.loadmat(
        "./measured_neurons/dabney_matlab/dabney_utility_fit.mat")
    zero_crossings = fig5['zeroCrossings_all'][:, 0]
    scaleFactNeg_all = fig5['scaleFactNeg_all'][:, 0]
    scaleFactPos_all = fig5['scaleFactPos_all'][:, 0]
    asymM_all = fig5['asymM_all'][:, 0]
    ZC_true_label = fig5['utilityAxis'].squeeze()

    def ZC_estimator(
        x): return fig5_betas["betas"][0, 0] + fig5_betas["betas"][1, 0] * x
    idx_to_maintain = np.where((scaleFactNeg_all * scaleFactPos_all) > 0)[0]
    asymM_all = asymM_all[idx_to_maintain]
    asymM_all_save = asymM_all.copy()
    idx_sorted = np.argsort(asymM_all)
    asymM_all = asymM_all[idx_sorted]
    estimated_ = np.array(ec_moment.get_quantiles_RPs(asymM_all))
    zero_crossings_ = fig5['zeroCrossings_all'][:, 0]
    zero_crossings_ = zero_crossings_[idx_to_maintain]
    zero_crossings_ = zero_crossings_[idx_sorted]
    zero_crossings_estimated = ZC_estimator(
        zero_crossings_)  # estimated thresholds

    fig, ax = plt.subplots(1, 1)
    # RPs = get_quantiles_RPs(ec_moment, quantiles_constant)
    RPs = ec_moment.get_quantiles_RPs(quantiles_constant)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.plot([0, 100], [0, 100], 'r')
    # ax.scatter(asymM_all, zero_crossings_estimated, s=10, color='k')
    ax.scatter(zero_crossings_estimated, asymM_all, s=10, color='k')
    # ax.scatter(quantiles_constant, thresholds_constant, s=20, color='#8856a7')
    ax.scatter(thresholds_constant, quantiles_constant, s=10, color='#9ebcda')
    # estimated_
    # zero_crossings_estimated
    # ax.scatter(estimated_, zero_crossings_estimated)
    # ax.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20], ['.1', '.3', '1.2', '2.5', '5', '10', '20'])
    ax.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20])
    # ax.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20, ], ['.1', '.3', '1.2', '2.5', '5', '10', '20'])
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 15])
    # ax.set_xlabel('Expected thresholds from asymmetry', fontsize=fontsize)
    # ax.set_ylabel('Thresholds $\\theta$', fontsize=fontsize)
    plt.grid(False)
    # fig.set_figwidth(6)
    # fig.set_figheight(4.5)
    fig.set_figwidth(4.5)
    fig.set_figheight(4.5)

    plt.savefig(
        'Fitting results asym2_default.pdf')
    plt.savefig(
        'Fitting results asym2_default')

    RPSS = ec_moment.get_quantiles_RPs(np.linspace(0, 1, 1000))
    # ax.plot(np.linspace(0,1,1000), RPSS, '--', color=[.7,.7,.7])
    ax.plot(RPSS, np.linspace(0, 1, 1000), '--', color=[.7, .7, .7])

    plt.savefig(
        'Fitting results asym2 dotted_default.pdf')

    import pickle as pkl
    import scipy.io as sio

    def sigmoid_func(x, a, b, c):
        return b / (1 + np.exp(-(x - c) * a))

    ec = ec_moment
    dir_measured_neurons = 'measured_neurons/'

    NDAT = sio.loadmat(dir_measured_neurons + 'data_max.mat')['dat']
    data = sio.loadmat(dir_measured_neurons + 'curve_fit_parameters.mat')

    indices = np.setdiff1d(np.linspace(0, 39, 40).astype(np.int16), 19)
    param_set = [data['ps_lcb'][indices], data['ps']
                 [indices], data['ps_ucb'][indices]]

    neurons_all = []
    for j in range(data['ps_total'][indices].shape[1]):
        neurons_all_ = []
        for i in range(len(ec.neurons_)):
            neurons_all_.append(sigmoid_func(
                ec.x, *data['ps_total'][indices][i, j, :]))
        neurons_all.append(neurons_all_)
    RSUM_all = [np.sum(np.array(neurons_all[i]) * ec.p_prior * ec._x_gap)
                for i in range(len(neurons_all))]

    N_neurons = 39

    # draw Figure
    fig2, ax2 = plt.subplots(1, 1)
    ax2.scatter(NDAT['ZC'][0, 0].squeeze(),
                param_set[1][:, 1], s=10, c=[0, 0, 0])

    alpha_lin = np.linspace(.2, 1, len(RSUM_all))
    rp1data = []
    rp2data = []

    R_t = 245.41
    ec = value_efficient_coding_moment(
        './', N_neurons=N_neurons, R_t=R_t, X_OPT_ALPH=0.7665, slope_scale=5.07)
    ec.replace_with_pseudo()

    ax2.plot(ec.x_inf, ec.g_x_pseudo, '-', linewidth=4,
             c='#9ebcda')

    g_x_rstar = []
    for i in range(len(ec.neurons_)):
        g_x_rstar.append(xss[id][0])
    tfff, quantiles_constant, thresholds_constant, alphas, xs, ys = ec.plot_approximate_kinky_fromsim_fitting_only_raw_rstar(
        ec.neurons_, '.', r_star_param=g_x_rstar, num_samples=int(1e4))

    # RPs = ec.get_quantiles_RPs(quantiles_constant)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 40)
    # ax.plot(RPs, ec.gsn, '-', marker = '|', linewidth=1, markersize=4,c='#8856a7')
    #
    # ax2.plot(RPs, ec.gsn_pseudo, '-', linewidth=4,
    #          c='#9ebcda' )
    fig.set_figwidth(4.5)
    fig.set_figheight(4.5)

    fig2.savefig(
        'RP2_data_90__default.png')
    fig2.savefig(
        'RP2_data_90__default.pdf')

    fig, ax = plt.subplots(1, 1)
    # RPs = get_quantiles_RPs(ec_moment, quantiles_constant)
    RPs = ec_moment.get_quantiles_RPs(quantiles_constant)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.plot([0, 100], [0, 100], 'r')
    # ax.scatter(asymM_all, zero_crossings_estimated, s=10, color='k')
    ax.scatter(zero_crossings_estimated, asymM_all, s=10, color='k')
    # ax.scatter(quantiles_constant, thresholds_constant, s=20, color='#8856a7')
    ax.scatter(NDAT['ZC'][0, 0].squeeze(),
               quantiles_constant, s=10, color='#9ebcda')
    # estimated_
    # zero_crossings_estimated
    # ax.scatter(estimated_, zero_crossings_estimated)
    # ax.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20], ['.1', '.3', '1.2', '2.5', '5', '10', '20'])
    ax.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20])
    # ax.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20, ], ['.1', '.3', '1.2', '2.5', '5', '10', '20'])
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 15])
    # ax.set_xlabel('Expected thresholds from asymmetry', fontsize=fontsize)
    # ax.set_ylabel('Thresholds $\\theta$', fontsize=fontsize)
    plt.grid(False)
    # fig.set_figwidth(6)
    # fig.set_figheight(4.5)
    fig.set_figwidth(4.5)
    fig.set_figheight(4.5)

    plt.savefig(
        'Fitting results asym22_default.pdf')
    plt.savefig(
        'Fitting results asym22_default')

    RPSS = ec_moment.get_quantiles_RPs(np.linspace(0, 1, 1000))
    # ax.plot(np.linspace(0,1,1000), RPSS, '--', color=[.7,.7,.7])
    ax.plot(RPSS, np.linspace(0, 1, 1000), '--', color=[.7, .7, .7])

    plt.savefig(
        'Fitting results asym2 dotted2_default.pdf')

    # draw Figure
    fig4, ax4 = plt.subplots(1, 1)
    idx_sorted_ = np.argsort(asymM_all_save)
    # ax4.scatter(zero_crossings_estimated,  param_set[1][:, 1][idx_sorted_], s=10, color=[0,0,0])
    from scipy.optimize import curve_fit

    hires_x = np.linspace(0, 15, 1000)
    def func(x, a, b): return a*x + b
    # best_fit_ab, covar = curve_fit(func, NDAT['ZC'][0,0].squeeze(), param_set[1][:,1],
    #                                absolute_sigma = True)
    best_fit_ab, covar = curve_fit(func, zero_crossings_estimated, param_set[1][:, 1][idx_sorted_],
                                   absolute_sigma=True)
    sigma_ab = np.sqrt(np.diagonal(covar))
    # tval = 1.96 # 95%
    tval = 1.66  # 90%
    bound_upper = func(hires_x, *(best_fit_ab + sigma_ab*tval))
    bound_lower = func(hires_x, *(best_fit_ab - sigma_ab*tval))
    scipy.stats.pearsonr(zero_crossings_estimated,
                         param_set[1][:, 1][idx_sorted_])
    # ax4.plot(hires_x, func(hires_x, *best_fit_ab), 'black')
    # ax4.fill_between(hires_x, bound_lower, bound_upper,
    #                  color = 'black', alpha = 0.15)

    alpha_lin = np.linspace(.2, 1, len(RSUM_all))
    rp1data = []
    rp2data = []
    for count, R in enumerate(np.sort(RSUM_all)):
        ec = value_efficient_coding_moment(
            './', N_neurons=N_neurons, R_t=R, X_OPT_ALPH=0.7665, slope_scale=5.07)
        ec.replace_with_pseudo()

        ax4.plot(ec.x_inf, ec.g_x_pseudo, '-', linewidth=1,
                 c='#9ebcda', alpha=1)

        g_x_rstar = []
        for i in range(len(ec.neurons_)):
            g_x_rstar.append(xss[id][0])
        print(count)

        ax4.set_xlim(0, 12)
        ax4.set_ylim(0, 40)
    # ax3.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20], ['.1', '.3', '1.2', '2.5', '5', '10', '20'])
    ax4.set_xticks([.1, .3, 1.2, 2.5, 5, 10, 20])

    ax4.scatter(zero_crossings_estimated,
                param_set[1][:, 1][idx_sorted_], s=10, c=[0, 0, 0])
    ax4.plot(hires_x, func(hires_x, *best_fit_ab), 'black')
    ax4.fill_between(hires_x, bound_lower, bound_upper,
                     color='black', alpha=0.15)
    fig4.set_figwidth(4.5)
    fig4.set_figheight(4.5)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)

    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 40)
    fig4.savefig(
        'RP4_data_90__default.png')
    fig4.savefig(
        'RP4_data_90__default.pdf')


if __name__ == "__main__":
    main()
    test()
