import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, uniform, gamma, lognorm
import os
import seaborn as sns
import time
import scipy.io as sio
import scipy
import pickle as pkl
# efficient coding using sigmoid response functions

class value_efficient_coding_moment():
    def __init__(self, prior='uniform', N_neurons=40, R_t = 250, X_OPT_ALPH=1.0, slope_scale = 4, simpler = False):
        # real data prior
        self.offset = 0
        self.juice_magnitudes = np.array(
            [0.2, 0.4, 0.6]) + self.offset
        if prior == 'uniform':
            self.juice_prob = np.array([1/3.0, 1/3.0, 1/3.0])
        else:
            self.juice_prob = np.array([2/15.0, 11/15.0, 2/15.0])
        self.juice_prob /= np.sum(self.juice_prob)
        self.juice_reward = [0.2,0.4,0.6]

        if simpler: # to boost computation
            self.x = np.linspace(0, 0.8, num=int(1e3))
            self.x_inf = np.linspace(0, 8, num=int(1e4))
        else:
            self.x = np.linspace(0, 0.8, num=int(1e4))
            self.x_inf = np.linspace(0, 8, num=int(1e5))

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 0.8]

        p_thresh = (2 * np.arange(N_neurons) + 1) / N_neurons / 2


        if prior == 'uniform':

            # Compute first two moments of the data
            mean_data = np.mean(self.juice_reward)
            var_data = np.var(self.juice_reward)

            # Compute parameters of the uniform distribution that matches the first two moments
            a = mean_data - np.sqrt(3 * var_data)
            b = mean_data + np.sqrt(3 * var_data)

            # Create the uniform distribution object
            uniform_dist = uniform(loc=a, scale=b - a)


            # self.p_prior = np.zeros( self.x.shape)
            # self.p_prior[2500:7500] = uniform.pdf(self.x[2500:7500]) # np.argmin(np.abs(self.x-0.2)), np.argmin(np.abs(self.x-0.6))
            self.p_prior = uniform_dist.pdf(self.x) # np.argmin(np.abs(self.x-0.2)), np.argmin(np.abs(self.x-0.6))
            # self.p_prior = lognorm.pdf(self.x, s=0.71, scale=np.exp(1.289))
            # self.p_prior_inf = np.zeros( self.x_inf.shape)
            # self.p_prior_inf[2500:7500] = uniform.pdf(self.x_inf[2500:7500]) #np.argmin(np.abs(self.x-0.2)), np.argmin(np.abs(self.x-0.6))
            self.p_prior_inf = uniform_dist.pdf(self.x_inf) # np.argmin(np.abs(self.x-0.2)), np.argmin(np.abs(self.x-0.6))
            # self.p_prior_inf = lognorm.pdf(self.x_inf, s=0.71, scale=np.exp(1.289))

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
        else:

            # Sample data with specific distribuion
            data = np.array([*(2 * [0.2]), *(11 * [0.4]), *(2 * [0.6])])

            # Compute first two moments of the data
            mean_data = np.mean(data)
            std_data = np.std(data)

            # Create the Gaussian distribution object
            norm_dist = norm(loc=mean_data, scale=std_data)

            self.p_prior= norm_dist.pdf(self.x)
            self.p_prior_inf = norm_dist.pdf(self.x_inf)

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
        self.cum_P = np.cumsum(p_prior_sum) # - 1e-3  # for approximation
        self.cum_P /= 1+1e-3

        # p_prior_inf_sum = self.p_prior_inf/np.sum(self.p_prior_inf)
        p_prior_inf_sum = self.p_prior_inf / np.sum(self.p_prior_inf)
        self.cum_P_pseudo = np.cumsum(
            p_prior_inf_sum) # - 1e-5  # for approximation
        self.cum_P_pseudo /= 1+1e-3

        norm_d = self.p_prior / (1-self.cum_P)**(1-X_OPT_ALPH)
        NRMLZR = np.sum(norm_d * self._x_gap)
        norm_d = norm_d / NRMLZR

        cum_norm_D = np.cumsum(self.N * norm_d * self._x_gap)
        cum_norm_Dp = np.cumsum(self.N * norm_d * self._x_gap)/cum_norm_D[-1]

        thresh_ = np.interp(p_thresh, cum_norm_Dp, self.x)
        quant_ = np.interp(thresh_, self.x, cum_norm_Dp)


        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g = self.p_prior / ((1 - self.cum_P)**X_OPT_ALPH)
        # norm_g /= NRMLZR
        norm_g /= self.N
        norm_g *= self.R

        norm_d_pseudo = self.p_prior_pseudo / \
            (1-self.cum_P_pseudo)**(1-X_OPT_ALPH)
        NRMLZR_pseudo = np.sum(norm_d_pseudo * self._x_gap)
        norm_d_pseudo = norm_d_pseudo / NRMLZR_pseudo

        cum_norm_D_pseudo = np.cumsum(self.N * norm_d_pseudo * self._x_gap)
        cum_norm_D_pseudop = np.cumsum(self.N * norm_d_pseudo * self._x_gap)/cum_norm_D_pseudo[-1]

        thresh_pseudo_ = np.interp(p_thresh, cum_norm_D_pseudop, self.x_inf)
        quant_pseudo_ = np.interp(thresh_pseudo_, self.x_inf, cum_norm_D_pseudop)

        norm_g_pseudo = self.p_prior_pseudo / \
            ((1 - self.cum_P_pseudo)**X_OPT_ALPH)
        norm_g_pseudo /= self.N
        norm_g_pseudo *= self.R

        # find each neuron's location
        # preferred response of each neuron. It is x=0 in the prototype sigmoid function (where y=0.5)
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
            self.neurons_pseudo_.append(g_sn * scipy.special.betainc(a, b, cum_norm_D_pseudop))
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
        plt.savefig('./' + name + '/' + 'Response functions of {0} neurons 300.png'.format(self.N))
        plt.xlim([0, 0.8])
        plt.ylim([0, 100])
        plt.savefig('./' + name + '/' + 'Response functions of {0} neurons.png'.format(self.N))

    def neuron_R_fit_timesG_fixedR_paramLog(self, XX):

        # bound
        if XX[0] <= 0.01:
            XX[0] = 0.01
        if XX[0] > 50:
            XX[0] = 50

        print(XX)

        if XX[1] == 1:
            name = 'uniform'
        else:
            name = 'normal'

        fit_ = value_efficient_coding_moment(name, N_neurons=self.fit_.N, R_t=self.fit_.R, X_OPT_ALPH = XX[1],
                                           slope_scale=4)
        fit_.replace_with_pseudo()

        res_max = []
        res_min = []
        for i in range(len(fit_.neurons_)):
            res_max.append(np.max(fit_.neurons_[i]))
            res_min.append(np.min(fit_.neurons_[i]))
        r_star = np.min(res_max)
        g_x_rstar = []
        for i in range(len(fit_.neurons_)):
            g_x_rstar.append(XX)

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
            check_2.append(np.all( [ len(fit_.x[:temp_threshold])>1, len(fit_.x[temp_threshold:])>1 ]) )
            check_3.append(int(num_samples * fit_.cum_P_pseudo[temp_threshold]))


        from scipy.optimize import curve_fit
        func = lambda rp, offset: lambda x, a: a * (x - rp) + offset


        if not np.all(check_):
            return 1e10  # just end here

        if not np.all(check_2):
            return 1e10  # just end here

        tof, quantiles_constant, thresholds_constant, alphas, xs, ys = fit_.plot_approximate_kinky_fromsim_fitting_only_raw_rstar(
             fit_.neurons_,
            self.dir_save_figures, r_star_param=g_x_rstar, num_samples= num_samples)

        if tof:
            RPs = self.get_quantiles_RPs(fit_, quantiles_constant)
            loss_rp = np.log(np.mean(np.sum((np.array(thresholds_constant) - np.array(RPs)) ** 2)))
            loss_1 = np.log(np.mean(np.sum((np.array(thresholds_constant)-np.array(self.Dabneys[1]))**2)))
            print(loss_1)
            return loss_1
        else:
            print(1e10)
            return 1e10
