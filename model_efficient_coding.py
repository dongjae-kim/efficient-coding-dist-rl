import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, uniform, gamma

# efficient coding using sigmoid response functions

class value_efficient_coding():
    def __init__(self, prior = 'normal',N_neurons = 20):

        # target prior
        if prior == 'normal':
            self.x = np.linspace(0,10,num=int(1e3))
            self.x_inf = np.linspace(0,1000,num=int(1e5))
            self.p_prior = (norm.pdf(self.x, loc = 5, scale= 1))
            self.p_prior_inf  = (norm.pdf(self.x_inf, loc = 5, scale= 1))
            self._x_gap = self.x[1]- self.x[0]
            self.x_minmax = [0,10]

        elif prior == 'gamma':
            self.x = np.linspace(0,10,num=int(1e3))
            self.x_inf = np.linspace(0,1000,num=int(1e5))
            self.p_prior = (gamma.pdf(self.x, 2))
            self.p_prior_inf  = (gamma.pdf(self.x_inf, 2))
            self._x_gap = self.x[1]- self.x[0]
            self.x_minmax = [0,10]

        elif prior=='uniform':
            self.x = np.linspace(0,10,num=int(1e3))
            self.x_inf = np.linspace(0,1000,num=int(1e5))
            self.p_prior = (uniform.pdf(self.x,scale = 5))
            self.p_prior_inf = (uniform.pdf(self.x_inf,scale = 5))
            self._x_gap = self.x[1]- self.x[0]
            self.x_minmax = [0,10]
        else:
            assert ValueError

        # since we posit a distribution ranged in [0,20] (mostly) we hypothesized that integral from -inf to +inf is same
        # as the integral from 0 to 20 in this toy example. From now on, we just calculated cumulative distribution using
        # self.x, which ranged from 0 to 20.
        # a prototype sigmoidal response curve
        self.h_s = lambda x: 1/(1+np.exp(x))

        # number of neurons
        self.N = N_neurons

        # total population response: mean of R spikes
        self.R = 1
        
        # p_prior_sum = self.p_prior/np.sum(self.p_prior)
        # self.cum_P = np.cumsum(p_prior_sum)

        # to prevent 0 on denominator in self.g
        p_prior_inf_sum = self.p_prior_inf/np.sum(self.p_prior_inf)
        self.cum_P = np.cumsum(p_prior_inf_sum)


        # density & gain
        self.d = lambda s: self.N*self.p_prior[s]
        self.g = lambda s: self.R/((self.N)*(1-self.cum_P[s]))

        # find each neuron's location
        self.sn = [] # preferred response of each neuron. It is x=0 in the prototype sigmoid function (where y=0.5)

        self.d_x = np.empty((0,))
        self.g_x = np.empty((0,))
        for j in range(len(self.x)):
            self.d_x = np.concatenate((self.d_x, np.array([self.d(j)]) ))
            self.g_x = np.concatenate((self.g_x, np.array([self.g(j)]) ))
            # self.d_x.append(self.d(i)) # based on the assumption that our domain ranged [0,20] is approximately same as [-inf, inf]


        self.D = np.cumsum(self.d_x*self._x_gap) # multiply _x_gap to approximate continuous integral.
        for i in range(self.N):
            # argmin
            i_argmin = np.argmin(np.abs(self.D - (i + 1)))
            # minimum with margin
            i_isclose0 = np.squeeze(np.where(np.isclose((self.D - (i+1)),0)))
            ind_set = [i_argmin, *i_isclose0.tolist()]
            self.sn.append(self.x[  np.min( ind_set )  ]) # take the minimum of two

        # each neurons response function
        self.neurons_ = []  # self.N number of neurons
        # from e.q. (4.2) in ganguli et al. (2014)
        h_prime = lambda s: np.exp(-s)/((1+np.exp(-s))**2) # first derivative of prototype sigmoid function

        g_sns = []
        x_gsns = []
        for i in range(self.N):
            g_sn = self.g(np.squeeze(np.where(self.x == self.sn[i])))
            hn_inner_integral = []
            for j in range(len(self.x)):
                hn_inner_integral.append(self.d_x[j]*h_prime(self.D[j]-i)*self._x_gap)
            h_n = g_sn * np.cumsum(hn_inner_integral)
            self.neurons_.append(h_n)
            g_sns.append(g_sn)
            x_gsns.append(self.sn[i])

    def plot_approximate_kinky(self,r_star = 0.02):
        plt.figure()
        colors = np.linspace(0, 0.7, self.N)
        quantiles = []
        for i in range(self.N-1): #excluded the last one since it is noisy
            (E_hn_prime_lower, E_hn_prime_higher, quantile), (x, y), theta_x = self.hn_approximate(i,r_star)
            quantiles.append(quantile)
            plt.plot(x, y, color=str(colors[i]))
        plt.ylim((0,.10))
        plt.xlim((0,4))
        plt.title('Approximated')
        plt.show()
        plt.figure()
        xlocs = np.linspace(1,self.N-1,self.N-1)
        plt.bar(xlocs, np.array(quantiles))
        for i, v in enumerate(np.array(quantiles)):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v,2)))
        plt.title('quantile for each neuron')

        return quantiles

    def hn_approximate(self,n,r_star = 0.02):
        hn = self.neurons_[n]
        theta_x = self.x[np.argmin(np.abs(hn - r_star))]
        theta = np.argmin(np.abs(hn - r_star))

        # lower than theta = hn^(-1)(r_star)
        inner_sigma_hn_prime = []
        for i in range(theta):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i+1] - hn[i]))
        E_hn_prime_lower = np.sum(inner_sigma_hn_prime)
        
        # higher than theta
        inner_sigma_hn_prime = []
        for i in range(theta,len(self.x)-1):
            inner_sigma_hn_prime.append(self.p_prior[i] * (hn[i+1] - hn[i]))
        E_hn_prime_higher = np.sum(inner_sigma_hn_prime)

        # plot it
        out_ = []
        for i in range(len(self.x)):
            if i < theta:
                out_.append(E_hn_prime_lower * (self.x[i]-theta_x)+r_star)
            else:
                out_.append(E_hn_prime_higher * (self.x[i]-theta_x)+r_star)

        return (E_hn_prime_lower, E_hn_prime_higher, E_hn_prime_higher/(E_hn_prime_higher+E_hn_prime_lower)), (self.x, np.array(out_)), theta_x

    def plot_neurons(self):
        # plot neurons response functions
        colors = np.linspace(0, 0.7, self.N)
        plt.figure()
        for i in range(self.N-1):
            plt.plot(self.x, self.neurons_[i], color=str(colors[i]))
        plt.ylim((0,round(np.max(self.neurons_[self.N-2]),1)))
        plt.title('Response functions of {0} neurons'.format(self.N-1))
        plt.show()

    def plot_others(self):
        plt.figure()
        plt.title('Prior distribution')
        plt.plot(self.x, self.p_prior)
        plt.figure()
        plt.title('Density function')
        plt.plot(self.x, self.d_x)
        plt.figure()
        plt.title('Gain function')
        plt.plot(self.x,self.g_x)

    # def

class value_dist_rl(value_efficient_coding):
    def TODO(self):
        return 0

def main():
    print('Initiailze efficient coding part')
    ec = value_efficient_coding('gamma')
    ec.plot_neurons()
    ec.plot_others()
    r_star = np.max(ec.neurons_[0])*.9
    ec.plot_approximate_kinky(r_star)




    print('Initiailze distributional reinforcement learning part')
    # value_dist_rl('gamma')


if __name__=="__main__":
    main()