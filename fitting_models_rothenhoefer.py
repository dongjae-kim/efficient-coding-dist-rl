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
        norm_g = 1 / ((1 - self.cum_P)**X_OPT_ALPH)
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

        norm_g_pseudo = 1 / \
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
# class value_efficient_coding_ro(value_efficient_coding_moment):
#     def plot_neurons(self, name='gamma'):
#         # plot neurons response functions
#         colors = np.linspace(0, 0.7, self.N)
#         plt.figure()
#         ymax = []
#         for i in range(self.N):
#             plt.plot(self.x, self.neurons_[i], color=str(colors[i]))
#             ymax.append(self.neurons_[i][1499])
#         # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
#         plt.title('Response functions of {0} neurons'.format(self.N))
#         if not os.path.exists('./' + name + '/'):
#             os.makedirs('./' + name + '/')
#         plt.savefig('./' + name + '/' + 'Response functions of {0} neurons 300.png'.format(self.N))
#         plt.xlim([0, 0.8])
#         plt.ylim([0, 100])
#         plt.savefig('./' + name + '/' + 'Response functions of {0} neurons.png'.format(self.N))
#
#     def plot_others(self, name='gamma'):
#
#         plt.figure()
#         plt.title('Prior distribution')
#         plt.plot(self.x, self.p_prior)
#         if not os.path.exists('./' + name + '/'):
#             os.makedirs('./' + name + '/')
#         plt.xlim([0, 0.8])
#         plt.ylim([0, 2])
#         plt.savefig('./' + name + '/' + 'Prior distribution 300.png')
#         plt.figure()
#         plt.title('Density function')
#         plt.plot(self.x, self.d_x)
#         plt.xlim([0, 0.8])
#         plt.ylim([0, 2])
#         plt.savefig('./' + name + '/' + 'Density function 300.png')
#         plt.figure()
#         plt.title('Gain function')
#         plt.plot(self.x, self.g_x)
#         plt.xlim([0, 0.8])
#         plt.ylim([0, 100])
#         plt.savefig('./' + name + '/' + 'Gain function 300.png')
#
#         plt.clf()
#
#         # plt.figure()
#         # plt.title('Prior distribution')
#         # plt.plot(self.x[:ind30], self.p_prior[:ind30])
#         # if not os.path.exists('./' + name + '/'):
#         #     os.makedirs('./' + name + '/')
#         # plt.savefig('./' + name + '/' + 'Prior distribution.png')
#         # plt.figure()
#         # plt.title('Density function')
#         # plt.plot(self.x[:ind30], self.d_x[:ind30])
#         # plt.savefig('./' + name + '/' + 'Density function.png')
#         # plt.figure()
#         # plt.title('Gain function')
#         # plt.plot(self.x[:ind30], self.g_x[:ind30])
#         # plt.savefig('./' + name + '/' + 'Gain function.png')

# class value_dist_rl(value_efficient_coding_moment):
#     def __init__(self, ec, alphas, quantiles, thresholds, r_star, N_neurons=40, dir_save='drl'):
#         self.dir_save = './' + dir_save + '/'
#
#         if not os.path.exists(self.dir_save):
#             os.makedirs(self.dir_save)
#         self.ec = ec
#
#         self.N = N_neurons
#
#         self.neurons_ = []
#         self.neurons_mixed_ = []
#
#         self.N_neurons = N_neurons
#         self.r_star = r_star
#
#         self.alphas = alphas
#         # multiply something to make it done
#         # self.alphas *= 100
#         self.quantiles = quantiles
#         self.thresholds = thresholds
#         self.thresholds_mixed = thresholds
#
#         self.offset = 0
#         self.juice_magnitudes = np.array([.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
#         self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
#                                     0.31159175, 0.1509519,
#                                     0.07705306])  # it is borrowed from the original data of Dabney's
#         self.juice_prob /= np.sum(self.juice_prob)
#         self.x = np.linspace(0, 30, num=int(1e3))
#         self.x_inf = np.linspace(0, 300, num=int(1e4))
#         self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
#         self.x_log_inf = np.log(self.x_inf)  # np.linspace(-50, 50, num=int(1e4))
#
#         self._x_gap = self.x[1] - self.x[0]
#         self.x_minmax = [0, 21]
#
#         # sum_pdf = np.zeros(self.x.shape)
#         sum_pdf = np.ones(self.x_log.shape) * 1e-8
#         for i in range(len(self.juice_prob)):
#             temp_ = norm.pdf(self.x_log, np.log(self.juice_magnitudes[i]), .2)
#             sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]
#
#         self.p_prior = sum_pdf / np.sum(sum_pdf * self._x_gap)
#
#         # sum_pdf = np.zeros(self.x_inf.shape)
#         sum_pdf = np.ones(self.x_log_inf.shape) * 1e-10
#         for i in range(len(self.juice_prob)):
#             temp_ = norm.pdf(self.x_log_inf, np.log(self.juice_magnitudes[i]), .2)
#             sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]
#
#         self.p_prior_inf = sum_pdf / np.sum(sum_pdf * self._x_gap)
#
#         import pickle as pkl
#         with open('lognormal_params.pkl', 'rb') as f:
#             param = pkl.load(f)
#         mu_mle = param['mu_mle']
#         sigma_mle = param['sigma_mle']
#
#         self.p_prior = lognorm.pdf(self.x, s=sigma_mle, scale=np.exp(mu_mle))
#         self.p_prior_inf = lognorm.pdf(self.x_inf, s=sigma_mle, scale=np.exp(mu_mle))
#
#         self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
#         self.p_prior_inf = self.p_prior_inf / np.sum(self.p_prior_inf * self._x_gap)
#
#         self.p_prior_cdf = np.cumsum(self.p_prior)
#         self.p_prior_cdf[-1] = 1 + 1e-16
#
#         # self.x = np.linspace(0, 10, num=int(1e3))
#         # self.x_inf = np.linspace(0, 1000, num=int(1e5))
#         # self.p_prior = (gamma.pdf(self.x, 2))
#         # self.p_prior_cdf = np.cumsum(self.p_prior)
#         # self.p_prior_cdf[-1] = 1 + 1e-16
#         # self.p_prior_inf = (gamma.pdf(self.x_inf, 2))
#         self._x_gap = self.x[1] - self.x[0]
#         self.x_minmax = [0, 1]
#
#         for i in range(N_neurons):
#             each_neuron = []
#             for j in range(len(self.x)):
#                 # thresholds[i]
#                 if self.x[j] < thresholds[i]:
#                     # slope * (x - threshold) + r_star
#                     each_neuron.append(alphas[i][0] * (self.x[j] - thresholds[i]) + r_star)
#                 else:
#                     each_neuron.append(alphas[i][1] * (self.x[j] - thresholds[i]) + r_star)
#             # each_neuron_array = np.array(each_neuron)
#             # each_neuron_array[np.where(each_neuron>=np.max(self.ec.neurons_[i]))] = np.max(self.ec.neurons_[i])
#             # each_neuron = each_neuron_array.tolist()
#             self.neurons_.append([each_neuron])
#         np.random.seed(2021)
#         index_ = np.linspace(0, N_neurons - 1, N_neurons).astype(np.int16)
#         index = np.random.permutation(index_)
#
#         self.alphas_mixed = []
#         self.quantiles_mixed = []
#         self.thresholds_mixed = []
#         for i in range(N_neurons):
#             self.neurons_mixed_.append(self.neurons_[index[i]])
#             self.alphas_mixed.append(self.alphas[index[i]])
#             self.quantiles_mixed.append(self.quantiles[index[i]])
#             self.thresholds_mixed.append(self.thresholds[index[i]])
#         pick_n_neurons = np.linspace(0, len(self.x) - 1, len(self.x)).astype(int)
#         pick_n_neurons = np.random.permutation(pick_n_neurons)
#         self.thresholds_mixed = self.x[pick_n_neurons[:self.N]]
#
#     def init_figures(self, iternum):
#         colors = np.linspace(0, 0.7, self.N)
#         plt.figure(1)
#         plt.close()
#         plt.figure(1)
#         for i in range(self.N):
#             plt.plot(self.x, self.neurons_[i][0], color=str(colors[i]))
#         # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
#         plt.title('Response functions of {0} neurons'.format(self.N))
#         # plt.show()
#         plt.savefig(self.dir_save + 'Response Function_{0:04d}.png'.format(iternum))
#
#         plt.figure(2)
#         plt.close()
#         plt.figure(2)
#         for i in range(self.N):
#             plt.plot(self.x, self.neurons_mixed_[i][0], color=str(colors[i]))
#         # plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
#         plt.title('Response functions of mixed {0} neurons'.format(self.N))
#         # plt.show()
#         plt.savefig(self.dir_save + 'Mixed Response Function_{0:04d}.png'.format(iternum))
#
#     def sample_value_cal_prediction_error(self, num_samples=100):
#         # sampling values
#         sample_x = []
#         sample_x_idx = []
#         num_samples = num_samples
#         for s in range(num_samples):
#             sample_x.append(np.random.choice(self.x, p=self.p_prior / np.sum(self.p_prior)))
#             sample_x_idx.append(np.where(self.x == sample_x[-1])[0][0])
#         self.sample_x = sample_x
#         self.sample_x_idx = sample_x_idx
#
#         # neurons
#         E_PE = []
#         PEs = []
#         DAs = []
#         for i in range(self.N_neurons):
#             PE_each_neuron = []
#             DA_each_neuron = []
#             for j in range(num_samples):
#                 # y: self.neurons_[i][0][sample_x_idx[j]]
#                 PE_each_neuron.append(self.x[sample_x_idx[j]] - self.thresholds[i])
#                 if self.x[sample_x_idx[j]] < self.thresholds[i]:  # negative prediction error
#                     DA_each_neuron.append(self.alphas[i][0] * (self.x[sample_x_idx[j]] - self.thresholds[i]))
#                 else:
#                     DA_each_neuron.append(self.alphas[i][1] * (self.x[sample_x_idx[j]] - self.thresholds[i]))
#             PEs.append(PE_each_neuron)
#             DAs.append(DA_each_neuron)
#             E_PE.append(np.mean(PE_each_neuron))
#         self.PEs = PEs
#         self.DAs = DAs
#
#         # neurons_mixed
#         E_PE_mixed = []
#         PEs_mixed = []
#         DAs_mixed = []
#         for i in range(self.N_neurons):
#             PE_each_neuron = []
#             DA_each_neuron = []
#             for j in range(num_samples):
#                 # y: self.neurons_[i][0][sample_x_idx[j]]
#                 PE_each_neuron.append(self.x[sample_x_idx[j]] - self.thresholds_mixed[i])
#                 if self.x[sample_x_idx[j]] < self.thresholds_mixed[i]:
#                     DA_each_neuron.append(self.alphas[i][0] * (self.x[sample_x_idx[j]] - self.thresholds_mixed[i]))
#                 else:
#                     DA_each_neuron.append(self.alphas[i][1] * (self.x[sample_x_idx[j]] - self.thresholds_mixed[i]))
#             PEs_mixed.append(PE_each_neuron)
#             DAs_mixed.append(DA_each_neuron)
#             E_PE_mixed.append(np.mean(PE_each_neuron))
#         self.PEs_mixed = PEs_mixed
#         self.DAs_mixed = DAs_mixed
#
#         return E_PE, E_PE_mixed
#
#     def update_neurons(self):
#         # update neurons
#         self.neurons_ = []
#         for i in range(self.N):
#             each_neuron = []
#             for j in range(len(self.x)):
#                 # thresholds[i]
#                 if self.x[j] < self.thresholds[i]:
#                     # slope * (x - threshold) + r_star
#                     each_neuron.append(self.alphas[i][0] * (self.x[j] - self.thresholds[i]) + self.r_star)
#                 else:
#                     each_neuron.append(self.alphas[i][1] * (self.x[j] - self.thresholds[i]) + self.r_star)
#                 if np.any(np.isnan(np.array(each_neuron[-1]))):
#                     print('a')
#             # each_neuron_array = np.array(each_neuron)
#             # each_neuron_array[np.where(each_neuron>=np.max(self.ec.neurons_[i]))] = np.max(self.ec.neurons_[i])
#             # each_neuron = each_neuron_array.tolist()
#             self.neurons_.append([each_neuron])
#
#         # update neurons
#         self.neurons_mixed_ = []
#         for i in range(self.N):
#             each_neuron = []
#             for j in range(len(self.x)):
#                 # thresholds[i]
#                 if self.x[j] < self.thresholds_mixed[i]:
#                     # slope * (x - threshold) + r_star
#                     each_neuron.append(self.alphas[i][0] * (self.x[j] - self.thresholds_mixed[i]) + self.r_star)
#                 else:
#                     each_neuron.append(self.alphas[i][1] * (self.x[j] - self.thresholds_mixed[i]) + self.r_star)
#             # each_neuron_array = np.array(each_neuron)
#             # each_neuron_array[np.where(each_neuron>=np.max(self.ec.neurons_[i]))] = np.max(self.ec.neurons_[i])
#             # each_neuron = each_neuron_array.tolist()
#             self.neurons_mixed_.append([each_neuron])
#
#     def update_pes_using_sample(self):
#         # For each i-th neuron update using expected DA
#         self.thresholds_prev = np.copy(self.thresholds)
#         self.thresholds_mixed_prev = np.copy(self.thresholds_mixed)
#
#         # neurons
#         for i in range(self.N_neurons):
#             # self.thresholds[i] -= np.mean(self.DAs[i])
#             self.thresholds[i] += np.mean(self.DAs[i])
#             if self.thresholds[i] < 0:
#                 self.thresholds[i] = 0
#         # neurons_mixed
#         for i in range(self.N_neurons):
#             # self.thresholds_mixed[i] -= np.mean(self.DAs_mixed[i])
#             self.thresholds_mixed[i] += np.mean(self.DAs_mixed[i])
#             if self.thresholds_mixed[i] < 0:
#                 self.thresholds_mixed[i] = 0
#
#         # print('')
#
#     def cal_cdfs_pdfs(self, iternum):
#         # self.alphas # what contains the slopes.
#         # self.thresholds# the value each neuron is coding.
#
#         reversal_points = np.array(self.thresholds)
#         alphas = np.array(self.alphas)
#         idx = np.argsort(reversal_points)
#         gotcdf = []
#         gotcdf_x = []
#         xi = 0
#         for i in range(self.N_neurons):
#             x = self.x[xi]
#             while (x < reversal_points[idx[i]]):
#                 if (x > np.max(self.x)):
#                     break
#                 if xi >= len(self.x) - 1:
#                     break
#                 gotcdf.append(alphas[idx[i], 1] / np.sum(alphas[idx[i], :]))
#                 gotcdf_x.append(xi)
#                 xi += 1
#                 x = self.x[xi]
#         gotcdf = np.array(gotcdf)
#         gotcdf_x = np.array(gotcdf_x)
#         plt.figure()
#         plt.plot(gotcdf_x / 100, gotcdf)
#         plt.xlim((0, 10))
#         # plt.xticks(np.arange(0,10.1,step=2), ['0','2','4','6','8','10'])
#         plt.xticks(np.arange(0, 10.1, step=2))
#         plt.ylim((0, 1))
#         plt.savefig(self.dir_save + 'CDF_{0:04d}.png'.format(iternum))
#
#         plt.figure()
#         gotpdf = np.ediff1d(gotcdf, to_begin=gotcdf[0])
#         plt.plot(gotcdf_x / 100, gotpdf)
#         plt.xlim((0, 10))
#         plt.xticks(np.arange(0, 10.1, step=2))
#         plt.ylim((0, 1))
#         plt.savefig(self.dir_save + 'PDF_{0:04d}.png'.format(iternum))
#
#         reversal_points_mixed = np.array(self.thresholds_mixed)
#         alphas_mixed = np.array(self.alphas)
#         idx = np.argsort(reversal_points_mixed)
#         gotcdf_mixed = []
#         gotcdf_mixed_x = []
#         xi = 0
#         for i in range(self.N_neurons):
#             x = self.x[xi]
#             while (x < reversal_points_mixed[idx[i]]):
#                 if (x > np.max(self.x)):
#                     break
#                 if xi >= len(self.x) - 1:
#                     break
#                 gotcdf_mixed.append(alphas_mixed[idx[i], 1] / np.sum(alphas_mixed[idx[i], :]))
#                 gotcdf_mixed_x.append(xi)
#                 xi += 1
#                 x = self.x[xi]
#         gotcdf_mixed = np.array(gotcdf_mixed)
#         gotcdf_mixed_x = np.array(gotcdf_mixed_x)
#         plt.figure()
#         plt.plot(gotcdf_mixed_x / 100, gotcdf_mixed)
#         plt.xlim((0, 10))
#         # plt.xticks(np.arange(0,10.1,step=2), ['0','2','4','6','8','10'])
#         plt.xticks(np.arange(0, 10.1, step=2))
#         plt.ylim((0, 1))
#         # plt.show()
#         plt.savefig(self.dir_save + 'MIXED CDF_{0:04d}.png'.format(iternum))
#
#         plt.figure()
#         gotpdf_mixed = np.ediff1d(gotcdf_mixed, to_begin=gotcdf_mixed[0])
#         plt.plot(gotcdf_mixed_x / 100, gotpdf_mixed)
#         plt.xlim((0, 10))
#         plt.xticks(np.arange(0, 10.1, step=2))
#         # plt.ylim((0,1))
#         plt.savefig(self.dir_save + 'MIXED PDF_{0:04d}.png'.format(iternum))
#
#         # erasing wrong interpretations? later
#
#     def plot_thresholds(self, iternum):
#         plt.figure(998)
#         plt.close()
#         plt.figure(998)
#         sns.set_style('whitegrid')
#         sns.kdeplot(self.thresholds, bw=.75, color='k', lw=3., shade=True)
#         sns.rugplot(self.thresholds, color='k')
#         plt.xlim((0, 10))
#         plt.ylim((0, .4))
#         box_prop = dict(facecolor="wheat", alpha=1)
#         plt.text(6, 0.1, 'mean:' + str(np.mean(self.thresholds)), bbox=box_prop)
#         # plt.savefig('./drl/' + 'Value_{0:04d}.png'.format(iternum))
#         plt.savefig(self.dir_save + 'RPs_{0:04d}.png'.format(iternum))
#
#         plt.figure(999)
#         plt.close()
#         plt.figure(999)
#         sns.set_style('whitegrid')
#         sns.kdeplot(self.thresholds_mixed, bw=.75, color='k', lw=3., shade=True)
#         sns.rugplot(self.thresholds_mixed, color='k')
#         plt.xlim((0, 10))
#         plt.ylim((0, .4))
#         box_prop = dict(facecolor="wheat", alpha=1)
#         plt.text(6, 0.1, 'mean:' + str(np.mean(self.thresholds_mixed)), bbox=box_prop)
#         # plt.savefig('./drl/' + 'Mixed Value_{0:04d}.png'.format(iternum))
#         plt.savefig(self.dir_save + 'Mixed RPs_{0:04d}.png'.format(iternum))
#
#     def TODO(self):
#         return 0


# def get_If(neurons_, x):
#     If_ = []
#     x_gap = x[1] - x[0]
#     for i in range(1, len(x)):
#         inner = []
#         for n in range(len(neurons_)):
#             hn = neurons_[n]
#             if len(hn) == 1:
#                 hn = hn[0]
#             # if not np.any(np.isnan(hn)):
#             if hn[i] > 0:
#                 inner.append((((hn[i] - hn[i - 1]) / x_gap) ** 2) / (hn[i]))
#         If_.append(np.sum(inner))
#         if np.sum(inner) > 20:
#             print('why')
#     return x[1:], If_


# def neuron_neuron_offset(ec, neuron, start_offset, maxval=118.0502):
#     flag = 0
#     OFFSET = start_offset
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] + OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         OFFSET += 1
#     flag = 0
#     OFFSET = OFFSET - 2
#
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] + OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         OFFSET += 1e-1
#
#     flag = 0
#     OFFSET = OFFSET - 2e-1
#
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] + OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         OFFSET += 1e-2
#
#     flag = 0
#     OFFSET = OFFSET - 2e-2
#
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] + OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         OFFSET += 1e-4
#
#     flag = 0
#     OFFSET = OFFSET - 2e-4
#
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] + OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         OFFSET += 1e-5
#
#     neuneurons_ = []
#     for i in range(len(neuron)):
#         neuneurons_.append(
#             (neuron[i, :] + OFFSET).tolist())
#     neuneurons_ = np.array(neuneurons_)
#
#     return OFFSET, neuneurons_

# class value_efficient_coding_fitting_sd(value_efficient_coding_moment):
#     def __init__(self, prior='normal', N_neurons=18, R_t=247.0690, XX2=1.0):
#         # real data prior
#         self.offset = 0
#         self.juice_magnitudes = np.array([.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
#         self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
#                                     0.31159175, 0.1509519,
#                                     0.07705306])  # it is borrowed from the original data of Dabney's
#         self.juice_prob /= np.sum(self.juice_prob)
#
#         self.x = np.linspace(0, 30, num=int(1e3))
#         self.x_inf = np.linspace(0, 300, num=int(1e4))
#         self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
#         self.x_log_inf = np.log(self.x_inf)  # np.linspace(-50, 50, num=int(1e4))
#
#         self._x_gap = self.x[1] - self.x[0]
#         self.x_minmax = [0, 21]
#
#         # logarithm space
#         logmu = np.sum(np.log(self.juice_magnitudes) * self.juice_prob)
#         logsd = np.sqrt(np.sum(((np.log(self.juice_magnitudes) - logmu) ** 2) * self.juice_prob))
#
#         self.p_prior = lognorm.pdf(self.x, s=logsd, scale=np.exp(logmu))
#         self.p_prior_inf = lognorm.pdf(self.x_inf, s=logsd, scale=np.exp(logmu))
#
#         import pickle as pkl
#         with open('lognormal_params.pkl', 'rb') as f:
#             param = pkl.load(f)
#         mu_mle = param['mu_mle']
#         sigma_mle = param['sigma_mle']
#
#         # self.p_prior = lognorm.pdf(self.x,s = sigma_mle,scale= np.exp(mu_mle))
#         # self.p_prior_inf = lognorm.pdf(self.x_inf,s = sigma_mle,scale= np.exp(mu_mle))
#
#         # with open('empirical_lognormal.pkl', 'rb') as f:
#         #     param_emp = pkl.load(f)['param']
#         # self.p_prior = lognorm.pdf(self.x,s = param_emp[0], loc = param_emp[1],scale = param_emp[2])
#         # self.p_prior_inf = lognorm.pdf(self.x_inf,s = param_emp[0], loc = param_emp[1],scale = param_emp[2])
#
#         self.p_prior = lognorm.pdf(self.x, s=0.71, scale=np.exp(1.289))
#         self.p_prior_inf = lognorm.pdf(self.x_inf, s=0.71, scale=np.exp(1.289))
#
#         self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
#         self.p_prior_inf = self.p_prior_inf / np.sum(self.p_prior_inf * self._x_gap)
#
#         # p_prior_in_log = norm.pdf(self.x_log,log_mu,log_sd)
#         # self.p_prior = (p_prior_in_log) / np.sum(p_prior_in_log*self._x_gap)
#
#         # p_prior_in_log = norm.pdf(self.x_log_inf,log_mu,log_sd)
#         # self.p_prior_inf = p_prior_in_log / np.sum(p_prior_in_log*self._x_gap)
#         # self.p_prior = self.p_prior_inf[:1000]
#
#         # pseudo p-prior to make the sum of the p-prior in the range can be 1
#         self.p_prior_pseudo = []
#         ppp_cumsum = np.cumsum(self.p_prior_inf * self._x_gap)
#         ppp_cumsum /= ppp_cumsum[-1]  # Offset
#         self.p_prior_pseudo.append(ppp_cumsum[0])
#         for i in range(len(ppp_cumsum) - 1):
#             self.p_prior_pseudo.append((ppp_cumsum[i + 1] - ppp_cumsum[i]) / self._x_gap)
#         self.p_prior_pseudo = np.array(self.p_prior_pseudo)
#
#         # since we posit a distribution ranged in [0,20] (mostly) we hypothesized that integral from -inf to +inf is same
#         # as the integral from 0 to 20 in this toy example. From now on, we just calculated cumulative distribution using
#         # self.x, which ranged from 0 to 20.
#         # a prototype sigmoidal response curve
#         self.h_s = lambda x: 1 / (1 + np.exp(x))
#
#         # number of neurons
#         self.N = N_neurons
#
#         # total population response: mean of R spikes
#         self.R = R_t
#
#         # p_prior_sum = self.p_prior/np.sum(self.p_prior)
#         # self.cum_P = np.cumsum(p_prior_sum)
#
#         # to prevent 0 on denominator in self.g
#         p_prior_sum = self.p_prior / np.sum(self.p_prior)
#         self.cum_P = np.cumsum(p_prior_sum) - 1e-3  # for approximation
#         # p_prior_inf_sum = self.p_prior_inf/np.sum(self.p_prior_inf)
#         p_prior_inf_sum = self.p_prior_inf / np.sum(self.p_prior_inf)
#         self.cum_P_pseudo = np.cumsum(p_prior_inf_sum) - 1e-5  # for approximation
#
#         norm_d = self.p_prior ** XX2 / (1 - self.cum_P) ** (1 - XX2)
#         NRMLZR = np.sum(norm_d * self._x_gap)
#         norm_d = norm_d / NRMLZR
#
#         cum_norm_D = np.cumsum(self.N * norm_d * self._x_gap)
#
#         # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
#         norm_g = self.p_prior ** (1 - XX2) / ((1 - self.cum_P) ** XX2)
#         # norm_g /= NRMLZR
#         norm_g /= self.N
#         norm_g *= self.R
#
#         norm_d_pseudo = self.p_prior_pseudo ** XX2 / (1 - self.cum_P_pseudo) ** (1 - XX2)
#         NRMLZR_pseudo = np.sum(norm_d_pseudo * self._x_gap)
#         norm_d_pseudo = norm_d_pseudo / NRMLZR_pseudo
#
#         cum_norm_D_pseudo = np.cumsum(self.N * norm_d_pseudo * self._x_gap)
#
#         # norm_g = self.p_prior_inf**(1-XX2) * self.R / ((self.N) * (1 - self.cum_P_pseudo)**XX2)
#         norm_g_pseudo = self.p_prior_pseudo ** (1 - XX2) / ((1 - self.cum_P_pseudo) ** XX2)
#         # norm_g /= NRMLZR
#         norm_g_pseudo /= self.N
#         norm_g_pseudo *= self.R
#         #
#         # # density & gain
#         # self.d = lambda s: self.N * self.p_prior[s]
#         # self.d_pseudo = lambda s: self.N * self.p_prior_pseudo[s]
#         # self.g = lambda s: self.R / ((self.N) * (1 - self.cum_P[s]))
#         # self.g_pseudo = lambda s: self.R / ((self.N) * (1 - self.cum_P_pseudo[s]))
#         #
#         # self.d_x = np.empty((0,))
#         # self.d_x_pseudo = np.empty((0,))
#         # self.g_x = np.empty((0,))
#         # self.g_x_pseudo = np.empty((0,))
#         # for j in range(len(self.x)):
#         #     self.d_x = np.concatenate((self.d_x, np.array([self.d(j)])))
#         #     self.g_x = np.concatenate((self.g_x, np.array([self.g(j)])))
#         #     # self.d_x.append(self.d(i)) # based on the assumption that our domain ranged [0,20] is approximately same as [-inf, inf]
#         #
#         # for j in range(len(self.x_inf)):
#         #     self.d_x_pseudo = np.concatenate((self.d_x_pseudo, np.array([self.d_pseudo(j)])))
#         #     self.g_x_pseudo = np.concatenate((self.g_x_pseudo, np.array([self.g_pseudo(j)])))
#
#         # find each neuron's location
#         self.sn = []  # preferred response of each neuron. It is x=0 in the prototype sigmoid function (where y=0.5)
#         self.sn_pseudo = []
#
#         self.D_pseudo = []
#
#         # self.D = np.cumsum(self.d_x * self._x_gap)  # multiply _x_gap to approximate continuous integral.
#         # self.D_pseudo = np.cumsum(self.d_x_pseudo * self._x_gap)  # pseudo version of it
#         # # offset
#         # self.D_pseudo += 0
#
#         ind_sets = []
#         ind_sets_pseudo = []
#         for i in range(self.N):
#             ind_set = np.argmin(np.abs(np.round(cum_norm_D + .5) - (i + 1)))
#             # ind_set = np.argmin(np.abs((self.D+.1) - s(i + 1)))
#             self.sn.append(self.x[np.min(ind_set)])  # take the minimum of two
#
#             ind_sets.append(np.min(ind_set))
#
#             # ind_set = np.argmin(np.abs(np.round(self.D_pseudo+.1) - (i + 1)))
#             ind_set = np.argmin(np.abs((cum_norm_D_pseudo + .5) - (i + 1)))
#             # i_isclose0 = np.squeeze(np.where(np.isclose((self.D_pseudo - (i+1)),0)))
#             # ind_set = [i_argmin, *i_isclose0.tolist()]
#             self.sn_pseudo.append(self.x_inf[np.min(ind_set)])  # take the minimum of two
#
#             ind_sets_pseudo.append(np.min(ind_set))
#
#         # each neurons response function
#         self.neurons_ = []  # self.N number of neurons
#
#         # from e.q. (4.2) in ganguli et al. (2014)
#         h_prime = lambda s: np.exp(-s) / ((1 + np.exp(-s)) ** 2)  # first derivative of prototype sigmoid function
#
#         g_sns = []
#         x_gsns = []
#         self.gsn = []
#
#         # for j in range(len(self.x)):
#         #     # hn_inner_integral.append(self.d_x[j]*h_prime(self.D[j]-(i+1))*self._x_gap)
#         #     print(self.D[j] - (i))
#         locs = []
#         hn_primes = []
#         for i in range(self.N):
#
#             locs.append(np.squeeze(np.where(self.x == self.sn[i])))
#             g_sn = norm_g[np.squeeze(np.where(self.x == self.sn[i]))]
#             hn_inner_integral = []
#             for j in range(len(self.x)):
#                 # hn_inner_integral.append(self.d_x[j]*h_prime(self.D[j]-(i+1))*self._x_gap)
#                 # hn_inner_integral.append(self.d_x[j] * h_prime(self.D[j] - (i + 1)) * self._x_gap)
#                 hn_inner_integral.append(norm_d[j] * h_prime(cum_norm_D[j] - (i + 1)) * self._x_gap)
#             h_n = g_sn * np.cumsum(hn_inner_integral)
#             self.neurons_.append(h_n)
#             g_sns.append(g_sn)
#             x_gsns.append(self.sn[i])
#             self.gsn.append(g_sn)
#             # hn_primes.append(h_prime(self.D[j] - (i + 1)))
#             hn_primes.append(h_prime(cum_norm_D[j] - (i + 1)))
#
#         g_sns = []
#         x_gsns = []
#         self.neurons_pseudo_ = []  # pseudo
#         self.gsn_pseudo = []
#         for i in range(self.N):
#             # g_sn = self.g(np.squeeze(np.where(self.x == self.sn[i])))
#             g_sn = norm_g_pseudo[np.squeeze(np.where(self.x_inf == self.sn_pseudo[i]))]
#             hn_inner_integral = []
#             for j in range(len(self.x_inf)):
#                 # hn_inner_integral.append(self.d_x_pseudo[j] * h_prime(self.D_pseudo[j] - (i + 1)) * self._x_gap)
#                 hn_inner_integral.append(norm_d_pseudo[j] * h_prime(cum_norm_D_pseudo[j] - (i + 1)) * self._x_gap)
#             h_n = g_sn * np.cumsum(hn_inner_integral)
#             self.neurons_pseudo_.append(h_n)
#             g_sns.append(g_sn)
#             x_gsns.append(self.sn_pseudo[i])
#             self.gsn_pseudo.append(g_sn)

# def neuron_neuron_scale(ec, neuron, start_offset, maxval=118.0502):
#     flag = 0
#     OFFSET = start_offset
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] * OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         if flag >= maxval:
#             break
#         OFFSET += 1
#     flag = 0
#     OFFSET = OFFSET - 1
#
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] * OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         if flag >= maxval:
#             break
#         OFFSET += 1e-1
#
#     flag = 0
#     OFFSET = OFFSET - 1e-1
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] * OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         if flag >= maxval:
#             break
#         OFFSET += 1e-2
#
#     flag = 0
#     OFFSET = OFFSET - 1e-2
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] * OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         if flag >= maxval:
#             break
#         OFFSET += 1e-4
#
#     flag = 0
#     OFFSET = OFFSET - 1e-4
#     while flag < maxval:
#         neuneurons_ = []
#         for i in range(len(neuron)):
#             neuneurons_.append(
#                 (neuron[i, :] * OFFSET).tolist())
#         neuneurons_ = np.array(neuneurons_)
#
#         dx = ec.x_inf[1] - ec.x_inf[0]
#         R_estimated = []
#         for xi in range(len(neuneurons_[0])):
#             R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
#         flag = np.sum(R_estimated)
#         print(flag)
#         print(OFFSET)
#         if flag >= maxval:
#             break
#         OFFSET += 1e-5
#
#     neuneurons_ = []
#     for i in range(len(neuron)):
#         neuneurons_.append(
#             (neuron[i, :] + OFFSET).tolist())
#     neuneurons_ = np.array(neuneurons_)
#
#     return OFFSET, neuneurons_

class fitting_model_model():
    def __init__(self, dir_save_figures, samples_idx, fit, X_OPT_ALPH, slopes):
        self.dir_save_figures = dir_save_figures
        self.samples_idx = samples_idx
        self.fit_ = fit
        self.slopes = slopes
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
        if XX[0] > 50:
            XX[0] = 50

        print(XX)
        if self.dir_save_figures == 'uniform':
            ALPHA = 1
        elif self.dir_save_figures == 'normal':
            ALPHA = 0


        fit_ = value_efficient_coding_moment(self.dir_save_figures, N_neurons=self.fit_.N, R_t=self.fit_.R, X_OPT_ALPH = ALPHA,
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

    def neuron_R_fit_fixed_rstar(self, XX):
        fit_ = value_efficient_coding(N_neurons=39, R_t=XX)
        fit_.replace_with_pseudo()
        quantiles_constant, thresholds_constant, alphas, xs, ys = fit_.plot_approximate_kinky_fromsamples_fitting_only(
            self.samples_idx, fit_.neurons_,
            name=self.dir_save_figures, r_star_param=.8)
        RPs = self.get_quantiles_RPs(fit_, quantiles_constant)

        return np.mean((np.array(thresholds_constant) - np.array(RPs)) ** 2)





def test():
    # fitting efficient code
    # savedir = './res_fit_ro_/'
    # savedir = './res_fit_to_empirical2/'
    #
    # # load the distribution fit (find the alpha)
    # LSE = 99999
    # LSE_s = []
    # x_opt_s = []
    #
    # import pickle as pkl
    #
    # for ii in range(10):
    #     with open(savedir + 'res_fit{0}.pkl'.format(ii), 'rb') as f:
    #         data_1 = pkl.load(f)
    #
    #     for i in range(len(data_1['res_s'])):
    #         LSE_s.append(data_1['res_s'][i]['fun'])
    #         x_opt_s.append(data_1['res_s'][i]['x'])
    #         if LSE > data_1['res_s'][i]['fun']:
    #             LSE = data_1['res_s'][i]['fun']
    #             x_opt = data_1['res_s'][i]['x']

    # idxxx = np.argsort(LSE_s)
    # id = np.argmin(LSE_s)

    N_neurons = 40
    R_t = 250
    dir_save_figures = './'
    print('Initiailze efficient coding part')
    ec_norm = value_efficient_coding_ro('normal', N_neurons=N_neurons, R_t=R_t, X_OPT_ALPH= 0.77, slope_scale = 5.07)
    ec_norm.plot_neurons('normal')
    ec_norm.plot_others('normal')
    ec_unfm = value_efficient_coding_ro('uniform', N_neurons=N_neurons, R_t=R_t, X_OPT_ALPH= 0.77, slope_scale = 5.07)
    ec_unfm.plot_neurons('uniform')
    ec_unfm.plot_others('uniform')


    spon_act = 6 # arbitrary
    norm_res = []
    unfm_res = []
    for i in range(N_neurons):
        norm_neg = ec_norm.neurons_[i][2500]-spon_act
        norm_0 = ec_norm.neurons_[i][5000]-spon_act
        norm_pos = ec_norm.neurons_[i][7500]-spon_act
        norm_res.append([norm_neg, norm_0, norm_pos])

        unfm_neg = ec_unfm.neurons_[i][2500]-spon_act
        unfm_0 = ec_unfm.neurons_[i][5000]-spon_act
        unfm_pos = ec_unfm.neurons_[i][7500]-spon_act
        unfm_res.append([unfm_neg, unfm_0, unfm_pos])
    print(np.mean(np.array(unfm_res), axis=0))
    print(np.mean(np.array(norm_res), axis=0))

    fig=plt.figure()
    plt.figsize()
    plt.plot(unfm_res[16],color='r')
    plt.plot(norm_res[16],color='b')

    import seaborn as sns
    data = np.array(unfm_res)
    # 데이터 생성
    y = np.mean(data,axis=0)
    error = np.std(data,axis=0)

    # seaborn으로 errorbar plot 그리기
    sns.set(style="whitegrid")
    sns.lineplot(x=[0.2,0.4,0.6], y=y,color='r',label='uniform')
    plt.errorbar(x=[0.2,0.4,0.6], y=y, yerr=error, fmt='o', markersize=8, capsize=3, capthick=1,color='r')

    # plot 장식하기
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.title("Uniform")

    data = np.array(norm_res)
    # 데이터 생성
    y = np.mean(data,axis=0)
    error = np.std(data,axis=0)

    sns.lineplot(x=[0.2,0.4,0.6], y=y,color='b',label='normal')
    plt.errorbar(x=[0.2,0.4,0.6], y=y, yerr=error, fmt='o', markersize=8, capsize=3, capthick=1,color='b')
    # plot 장식하기
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.title("Normal")
    # plot 보여주기
    plt.savefig('./comparison.png')
    plt.show()

    print('')

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
            R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
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
            R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
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
            R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
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
            R_estimated.append(ec.p_prior_inf[xi] * np.sum(neuneurons_[:, xi]) * dx)
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
        self.juice_magnitudes = np.array([.1, .3, 1.2, 2.5, 5, 10, 20]) + self.offset
        self.juice_prob = np.array([0.06612594, 0.09090909, 0.14847358, 0.15489467,
                                    0.31159175, 0.1509519,
                                    0.07705306])  # it is borrowed from the original data of Dabney's
        self.juice_prob /= np.sum(self.juice_prob)
        self.x = np.linspace(0, 30, num=int(750))
        self.x_inf = np.linspace(0, 300, num=int(7500))
        self.x_log = np.log(self.x)  # np.linspace(-5, 5, num=int(1e3))
        self.x_log_inf = np.log(self.x_inf)  # np.linspace(-50, 50, num=int(1e4))

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
            temp_ = norm.pdf(self.x_log_inf, np.log(self.juice_magnitudes[i]), .2)
            sum_pdf += temp_ / np.max(temp_) * self.juice_prob[i]

        self.p_prior_inf = sum_pdf / np.sum(sum_pdf * self._x_gap)

        import pickle as pkl
        with open('lognormal_params.pkl', 'rb') as f:
            param = pkl.load(f)
        mu_mle = param['mu_mle']
        sigma_mle = param['sigma_mle']

        self.p_prior = lognorm.pdf(self.x, s=sigma_mle, scale=np.exp(mu_mle))
        self.p_prior_inf = lognorm.pdf(self.x_inf, s=sigma_mle, scale=np.exp(mu_mle))

        self.p_prior = self.p_prior / np.sum(self.p_prior * self._x_gap)
        self.p_prior_inf = self.p_prior_inf / np.sum(self.p_prior_inf * self._x_gap)

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
    # load all files in res_fit_to_empirical_rothenhoefer
    files = os.listdir('./res_fit_to_empirical_rothenhoefer_MSE_sort')
    files = [f for f in files if f.endswith('.pkl')]
    files.sort()

    # make empty list
    res = []
    # for every file in files|
    for f in files:
        # open file using pickle
        with open('./res_fit_to_empirical_rothenhoefer_MSE_sort/' + f, 'rb') as f:
            res.append(pkl.load(f))

    # ...and sorting in ascending order of fun
    res_sorted = sorted(res, key=lambda x: x.fun)

    # find that has lowest fun
    res = res[np.argmin([r.fun for r in res])]
    # print res
    print(res)

    # ec_norm and ec_unfm with the fitted parameters
    params = res.x

    N_neurons = 40
    alpha_norm = params[0]
    alpha_unfm =params[1]
    # others uses res.x
    slope_scale_norm = params[2]
    slope_scale_unfm = params[3]
    # R_t = 150
    R_t = params[4]
    spon_act = 5
    samples = 10000

    # cal slopes using spon_act

    ec_norm = value_efficient_coding_moment('normal', N_neurons=N_neurons, R_t=R_t, X_OPT_ALPH= alpha_norm, slope_scale = slope_scale_norm)
    ec_unfm = value_efficient_coding_moment('uniform', N_neurons=N_neurons, R_t=R_t, X_OPT_ALPH= alpha_unfm, slope_scale = slope_scale_unfm)

    print('x')



if __name__ == "__main__":
    test()