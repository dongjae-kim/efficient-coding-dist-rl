#!/usr/bin/env python
# coding: utf-8

# # Moment matching debug

# In[45]:


import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt


class value_efficient_coding_moment():
    def __init__(self, prior='normal', N_neurons=18, R_t=247.0690, XX2 = 1.0):
        # real data prior
        self.offset = 0
        self.juice_magnitudes = np.array([.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
                                    0.31159175, 0.1509519,
                                    0.07705306])  # it is borrowed from the original data of Dabney's
        self.juice_prob /= np.sum(self.juice_prob)

        self.x = np.linspace(0, 30, num=int(1e3))
        self.x_inf = np.linspace(0, 300, num=int(1e4))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        self.x_log_inf = np.log(self.x_inf)  # np.linspace(-50, 50, num=int(1e4))

        self._x_gap = self.x[1] - self.x[0]
        self.x_minmax = [0, 21]

        # logarithm space
        logmu = np.sum(np.log(self.juice_magnitudes) * self.juice_prob)
        logsd = np.sqrt(np.sum(((np.log(self.juice_magnitudes) - logmu) ** 2) * self.juice_prob))

        self.p_prior = lognorm.pdf(self.x, s=logsd, scale=np.exp(logmu))
        self.p_prior_inf = lognorm.pdf(self.x_inf, s=logsd, scale=np.exp(logmu))

        import pickle as pkl
        # with open('lognormal_params.pkl', 'rb') as f:
        #     param = pkl.load(f)
        # mu_mle = param['mu_mle']
        # sigma_mle = param['sigma_mle']

        # with open('empirical_lognormal.pkl', 'rb') as f:
        #     param_emp = pkl.load(f)['param']
        # self.p_prior = lognorm.pdf(self.x,s = param_emp[0], loc = param_emp[1],scale = param_emp[2])
        # self.p_prior_inf = lognorm.pdf(self.x_inf,s = param_emp[0], loc = param_emp[1],scale = param_emp[2])

        self.p_prior = lognorm.pdf(self.x, s=0.71, scale=np.exp(1.289))
        self.p_prior_inf = lognorm.pdf(self.x_inf, s=0.71, scale=np.exp(1.289))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / np.sum(self.p_prior_inf * self._x_gap)

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
            self.p_prior_pseudo.append((ppp_cumsum[i + 1] - ppp_cumsum[i]) / self._x_gap)
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
        self.cum_P_pseudo = np.cumsum(p_prior_inf_sum) - 1e-5 # for approximation

        norm_d = self.p_prior / (1-self.cum_P)**(1-XX2)
        NRMLZR = np.sum(norm_d * self._x_gap)
        norm_d = norm_d / NRMLZR

        cum_norm_D = np.cumsum(self.N * norm_d * self._x_gap)

        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g = self.p_prior / ((1 - self.cum_P)**XX2)
        # norm_g /= NRMLZR
        norm_g /= self.N
        norm_g *= self.R


        norm_d_pseudo = self.p_prior_pseudo / (1-self.cum_P_pseudo)**(1-XX2)
        NRMLZR_pseudo = np.sum(norm_d_pseudo * self._x_gap)
        norm_d_pseudo = norm_d_pseudo / NRMLZR_pseudo

        cum_norm_D_pseudo = np.cumsum(self.N * norm_d_pseudo * self._x_gap)

        # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
        norm_g_pseudo = self.p_prior_pseudo / ((1 - self.cum_P_pseudo)**XX2)
        # norm_g /= NRMLZR
        norm_g_pseudo /= self.N
        norm_g_pseudo *= self.R

        # find each neuron's location
        self.sn = []  # preferred response of each neuron. It is x=0 in the prototype sigmoid function (where y=0.5)
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
            self.sn_pseudo.append(self.x_inf[np.min(ind_set)])  # take the minimum of two

            ind_sets_pseudo.append(np.min(ind_set))

        # each neurons response function
        self.neurons_ = []  # self.N number of neurons

        # from e.q. (4.2) in ganguli et al. (2014)
        h_prime = lambda s: np.exp(-s) / ((1 + np.exp(-s)) ** 2)  # first derivative of prototype sigmoid function

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
                hn_inner_integral.append(self.N*norm_d[j] * h_prime(cum_norm_D[j] - (i + 1)) * self._x_gap)
            h_n = g_sn * np.cumsum(hn_inner_integral)
            self.neurons_.append(h_n)
            g_sns.append(g_sn)
            x_gsns.append(self.sn[i])
            self.gsn.append(g_sn)
            # hn_primes.append(h_prime(self.D[j] - (i + 1)))
            hn_primes.append(h_prime(cum_norm_D[j] - (i + 1)))

        # normalize afterward
        NRMLZR_G = self.R/np.sum(np.array(self.neurons_) * self.p_prior * self._x_gap)
        # neurons_arr=np.array(self.neurons_)*NRMLZR_G
        for i in range(len(self.neurons_)):
            self.neurons_[i] *= NRMLZR_G
            self.gsn[i] *= NRMLZR_G


        g_sns = []
        x_gsns = []
        self.neurons_pseudo_ = []  # pseudo
        self.gsn_pseudo = []
        for i in range(self.N):
            # g_sn = self.g(np.squeeze(np.where(self.x == self.sn[i])))
            g_sn = norm_g_pseudo[np.squeeze(np.where(self.x_inf == self.sn_pseudo[i]))]
            hn_inner_integral = []
            for j in range(len(self.x_inf)):
                # hn_inner_integral.append(self.d_x_pseudo[j] * h_prime(self.D_pseudo[j] - (i + 1)) * self._x_gap)
                hn_inner_integral.append(self.N*norm_d_pseudo[j] * h_prime(cum_norm_D_pseudo[j] - (i + 1)) * self._x_gap)
            h_n = g_sn * np.cumsum(hn_inner_integral)
            self.neurons_pseudo_.append(h_n)
            g_sns.append(g_sn)
            x_gsns.append(self.sn_pseudo[i])
            self.gsn_pseudo.append(g_sn)

        # normalize afterward
        NRMLZR_G_pseudo = self.R/np.sum(np.array(self.neurons_pseudo_) * self.p_prior_pseudo * self._x_gap)
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
            plt.plot(self.x_inf, self.neurons_pseudo_[i], color=str(colors[i]))
            ymax.append(self.neurons_pseudo_[i][1499])
        # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title('Response functions of {0} neurons'.format(self.N))
        plt.xlim([0, 20])
        plt.ylim([0, np.max(ymax)])


# ## Response functions

# In[49]:

#
R = 255.19
N = 39
ec=value_efficient_coding_moment(N_neurons=N, R_t=R, XX2=.9)
#
#
#
# get_ipython().run_line_magic('matplotlib', 'notebook')
# ec.plot_neurons()


# ## Import midpoints of neurons
# 

# In[90]:


import scipy.io as sio
from scipy.optimize import minimize
import os
data = sio.loadmat('measured_neurons/curve_fit_parameters.mat')['ps']
midpoints = data[np.setdiff1d(np.linspace(0, 39, 40).astype(np.int), 19), 2]
midpoints = np.sort(midpoints)


# In[92]:


def nll_cal(XX2):
    if XX2<0:
        XX2=0
    ec=value_efficient_coding_moment(N_neurons=N, R_t=R, XX2=XX2)
    idxx = [np.argmin(np.abs(ec.x_inf - midpoints[i])) for i in range((ec.N))]
    NLL = -np.sum(np.log((ec.d_x[idxx]+np.finfo(np.float32).eps)*ec._x_gap))
    print(XX2)
    print(NLL)
    if np.isnan(NLL):
        print(ec._x_gap)

    return NLL

num_seed = 10
XX22 = np.linspace(0,1,num_seed).tolist()
from itertools import product
import sys
max_total = 10

savedir = 'res_fit_to_empirical3/'

if not os.path.exists(savedir):
    os.makedirs(savedir)

for ii in range(max_total):
    if not os.path.exists(savedir+ 'res_fit{0}.pkl'.format(ii)):

        len_ind = int(len(XX22) / max_total)

        ind = np.linspace(len_ind * ii, len_ind * (ii + 1) - 1, len_ind).astype(np.int)

        print(ii)
        XXS = np.array(XX22)[ind].tolist()
        res_s = []
        for XX in XXS:
            res = minimize(nll_cal,XX, method='nelder-mead',
                   options={'maxiter': 1e5,  'disp': True})

            res_s.append(res)
        
        import pickle as pkl
        with open(savedir+ 'res_fit{0}.pkl'.format(ii),'wb') as f:
            pkl.dump({'res_s':res_s}, f)

