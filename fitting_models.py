import numpy as np
import pickle as pkl
from scipy.stats import lognorm
import os
import time
import scipy.io as sio
import scipy
from scipy.optimize import minimize, minimize_scalar

# efficient coding using sigmoid response functions

N_neurons = 39
R_t = 245.41
slope_scale = 5.07


def get_thresholds(alpha=1, r_star=5, slope_scale=5, N_neurons=39, R_t=245.41):
    """get threshold for each neuron without creating a whole object"""
    p_thresh = (2 * np.arange(N_neurons) + 1) / N_neurons / 2
    x = np.linspace(np.finfo(float).eps, 100, num=int(1e4))
    _x_gap = x[1] - x[0]
    p_prior = lognorm.pdf(x, s=0.71, scale=np.exp(1.289))
    p_prior = p_prior / np.sum(p_prior * _x_gap)
    cum_P = np.cumsum(p_prior)
    cum_P /= cum_P[-1] + 1e-4

    density = p_prior / (1 - cum_P) ** (1 - alpha)
    cum_d = np.cumsum(density)
    cum_d /= cum_d[-1] + 1e-4

    thresh_ = np.interp(p_thresh, cum_d, x)
    # compute response functions
    neurons = []  # self.N number of neurons
    norm_g = 1 / ((1 - cum_P) ** alpha)

    for i in range(N_neurons):
        g_sn = np.interp(thresh_[i], x, norm_g)

        a = slope_scale * p_thresh[i]
        b = slope_scale * (1 - p_thresh[i])
        neurons.append(g_sn * scipy.special.betainc(a, b, cum_d))

    neurons = np.array(neurons)
    # normalize afterward
    NRMLZR_G = R_t / np.sum(neurons * p_prior * _x_gap)
    neurons *= NRMLZR_G
    thresholds = []
    for i in range(N_neurons):
        thresholds.append(np.interp(r_star, neurons[i], x, left=0, right=100))
    return np.array(thresholds)


def squared_error_thresholds(
    true_thresh, alpha=1, r_star=5, slope_scale=5, N_neurons=39, R_t=245.41
):
    """XX = [alpha, r_star]?"""
    # bound
    if alpha <= 0.01 or alpha > 1:
        print(1e10)
        return 1e10
    # bound
    if r_star <= 0.01 or r_star > 50:
        # print(1e10)
        return 1e10

    thresholds = get_thresholds(
        alpha=alpha,
        r_star=r_star,
        slope_scale=slope_scale,
        N_neurons=N_neurons,
        R_t=R_t,
    )

    loss_1 = np.mean((np.sort(thresholds) - np.sort(np.array(true_thresh))) ** 2)
    # print(loss_1)
    return loss_1


def get_loss_r_star_slope(true_thresh, alpha=1):
    """pars=[r_star, slope]"""

    def loss(pars):
        return squared_error_thresholds(
            true_thresh, r_star=pars[0], slope_scale=pars[1], alpha=alpha
        )

    return loss


def log_density_ec(midpoints, alpha=1, s=0.71, scale=np.exp(1.289)):
    """log-density of midpoints according to the efficient code
    for fitting we actually remove the log-pdf part that is not changed by alpha
    """
    # log_d = (
    #    lognorm.logpdf(midpoints, s=s, scale=scale)
    #    - (1 - alpha) * np.log(1 - lognorm.cdf(midpoints))
    #    + np.log(alpha)
    # )
    log_d = np.log(alpha) - (1 - alpha) * np.log(
        1 - lognorm.cdf(midpoints, s=s, scale=scale)
    )
    return log_d


def get_loss_alpha(midpoints):
    def loss(alpha):
        if alpha <= 0:
            return 1e10
        else:
            return -np.sum(log_density_ec(midpoints, alpha=alpha))

    return loss


def fit_alpha(alpha_dir="res_alpha"):
    data = sio.loadmat("curve_fit_parameters.mat")["ps"]
    midpoints = data[np.setdiff1d(np.linspace(0, 39, 40).astype(int), 19), 2]
    loss = get_loss_alpha(midpoints)
    if not os.path.exists(alpha_dir):
        os.makedirs(alpha_dir)
    # this is a 1D convex function, no multiple starts needed
    res = minimize_scalar(loss, bounds=[0, 1])
    with open(os.path.join(alpha_dir + "res_alpha.pkl"), "wb") as f:
        pkl.dump({"res": res}, f)


def fit_rstar_slope(alpha_dir="res_alpha", savedir="res_rstar_slope/"):
    # fitting thresholds by adjusting overall slope and r_star

    # load alpha
    with open(os.path.join(alpha_dir, "res_alpha.pkl"), "rb") as f:
        data = pkl.load(f)
        alpha = data["res"].x

    # load thresholds
    NDAT = sio.loadmat(os.path.join("measured_neurons", "data_max.mat"))["dat"]
    true_thresh = NDAT["ZC"][0, 0].squeeze()

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # choose starting paramters using meshgrid
    num_seed = 5
    XX0 = np.linspace(1.5, 15, num_seed)
    XX1 = np.linspace(0.5, 10, num_seed)
    XX = np.array(np.meshgrid(XX0, XX1)).T.reshape(-1, 2)

    loss = get_loss_r_star_slope(true_thresh, alpha)

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
    fit_alpha()
    fit_rstar_slope()
