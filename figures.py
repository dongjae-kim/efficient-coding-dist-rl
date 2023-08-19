#!/usr/bin/env python
# coding: utf-8

# before running this file run:
# $ python fit_sigmoids.py
# $ python fitting_models.py

import os
import argparse
import pickle as pkl
import numpy as np
import scipy.io as sio
import scipy
from scipy import stats
from scipy.stats import pearsonr
import scipy.signal
import matplotlib.pyplot as plt
from ec_class import value_efficient_coding_moment, juice_magnitudes, juice_prob
from illustration_efficient import plot_optimal, make_solution, get_pars, make_efficient

# local function definitions

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-g", "--gamma", action="store_true")
    args = parse.parse_args()
    use_gamma = args.gamma
else:
    use_gamma = False

if use_gamma:
    fig_folder = "figures_gamma"
else:
    fig_folder = "figures"

N_neurons = 39
dir_measured_neurons = "measured_neurons/"


def sigmoid_func(x, a, b, c):
    return b / (1 + np.exp(-(x - c) * a))


def first_derivative_sigmoid_func(x, a, b, c):
    # first derivative of the sigmoid function
    return a * b * np.exp(-a * (x - c)) / (1 + np.exp(-a * (x - c))) ** 2


def log_kde(x, y, sigma, x_sample):
    """calculates a log-normal based kernel density estimate
    based on samples at x with weights/probabilities y evaluated at x_sample"""
    # mean = exp(mu + sigma^2/2) != mtrue
    # -> mu = log(mtrue) - sigma^2/2
    mu = np.log(x) - (sigma**2 / 2)
    y_sample = np.zeros_like(x_sample)
    for mu_i, y_i in zip(mu, y):
        y_sample += (
            np.exp(-((np.log(x_sample) - mu_i) ** 2) / 2 / (sigma**2))
            / np.sqrt(2 * np.pi)
            / sigma
            / x_sample
        ) * y_i
    return y_sample


# load fitted alpha
if use_gamma:
    alpha_dir = "res_alpha_gamma"
    savedir = "res_rstar_slope_rt_gamma"
else:
    alpha_dir = "res_alpha"
    savedir = "res_rstar_slope_rt"
with open(os.path.join(alpha_dir, "res_alpha.pkl"), "rb") as f:
    data = pkl.load(f)
    alpha = data["res"].x

# load fitted r_star and slope_scale
pars = []
loss = []
for i in range(25):
    fname = os.path.join(savedir, "res_rstar_slope_{0}.pkl".format(i))
    if os.path.exists(fname):
        with open(os.path.join(fname), "rb") as f:
            data = pkl.load(f)
            pars.append(data["res"].x)
            loss.append(data["res"].fun)
idx_best = np.argmin(loss)
pars = pars[idx_best]
r_star = pars[0]
slope_scale = pars[1]
R_t = pars[2]

# parameter fitted
print(alpha)
print(R_t)
print(r_star)
print(slope_scale)

ec = value_efficient_coding_moment(
    N_neurons=N_neurons,
    R_t=R_t,
    alpha=alpha,
    slope_scale=slope_scale,
    use_gamma=use_gamma,
)
ec_simple = value_efficient_coding_moment(
    N_neurons=N_neurons,
    R_t=R_t,
    alpha=alpha,
    slope_scale=slope_scale,
    simpler=True,
    use_gamma=use_gamma,
)
ec_simple.replace_with_pseudo()
NDAT = sio.loadmat(dir_measured_neurons + "data_max.mat")["dat"]
data = sio.loadmat("curve_fit_parameters_min.mat")

indices = np.setdiff1d(np.linspace(0, 39, 40).astype(np.int16), 19)  # except one neuron
param_set = data["ps"][indices]
midpoints = data["ps"][indices, 2]

# get asymmatric slopes
# increase num_samples to int(1e4)
(
    tf,
    quantiles_ec,
    thresh_ec,
    alphas,
    xs,
    ys,
) = ec.plot_approximate_kinky_fromsim_fitting_only_raw_rstar(
    r_star=r_star, num_samples=int(1e4)
)

# load details from Dabneys paper. refer to `./measured_neurons/dabney_matlab/`
fig5 = sio.loadmat("./measured_neurons/dabney_matlab/dabney_fit.mat")
fig5_betas = sio.loadmat("./measured_neurons/dabney_matlab/dabney_utility_fit.mat")


def ZC_estimator(x):
    return fig5_betas["betas"][0, 0] + fig5_betas["betas"][1, 0] * x


scaleFactNeg_all = fig5["scaleFactNeg_all"][:, 0]
scaleFactPos_all = fig5["scaleFactPos_all"][:, 0]
asymM_all = fig5["asymM_all"][:, 0]
ZC_true_label = fig5["utilityAxis"].squeeze()
idx_to_maintain = np.where((scaleFactNeg_all * scaleFactPos_all) > 0)[0]
asymM_all = asymM_all[idx_to_maintain]
asymM_all_save = asymM_all.copy()
idx_sorted = np.argsort(asymM_all)
asymM_all = asymM_all[idx_sorted]
estimated_ = np.array(ec.get_quantiles_RPs(asymM_all))
zero_crossings = fig5["zeroCrossings_all"][:, 0]
zero_crossings = zero_crossings[idx_to_maintain]
zero_crossings = zero_crossings[idx_sorted]
zero_crossings_estimated = ZC_estimator(zero_crossings)  # estimated thresholds
true_thresh = sio.loadmat(os.path.join(
    "measured_neurons", "data_max.mat")
    )["dat"]["ZC"][0, 0].squeeze()

# Midpoints figure
ys = log_kde(midpoints, np.ones(39) / 39, 0.6, ec.x)
ys *= np.sum(ec.p_prior) / np.sum(ys)

fig, ax = plt.subplots(1, 1)
# Reward distribution
plt.plot(ec.x, ec.p_prior, color="#999999")
# Efficient code density
plt.plot(ec.x, ec.d_x, color="#2D59B3")
# kde estimate of real neurons
plt.plot(ec.x, ys, color="k")
# individual neurons
ax.vlines(midpoints, 0, 0.012 * np.ones(midpoints.shape), colors="k")
# Decoration
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(0, 25)
ax.set_ylim(0, 0.22)
ax.set_xticks([0, 10, 20])
fig.set_figwidth(6.2)
fig.set_figheight(2)
plt.legend(
    ["reward density", "efficient code", "measured neurons"], fontsize=14, frameon=False
)
plt.xlabel("Midpoints", fontdict={"size": 16})

plt.savefig(os.path.join(".", fig_folder, "midpoints.png"), bbox_inches="tight")
plt.savefig(os.path.join(".", fig_folder, "midpoints.pdf"), bbox_inches="tight")

# Thresholds figure
ys = log_kde(true_thresh, np.ones(39) / 39, 0.5, ec.x)
ys *= np.sum(ec.p_prior) / np.sum(ys)
ys_ec = log_kde(thresh_ec, np.ones(39) / 39, 0.5, ec.x)
ys_ec *= np.sum(ec.p_prior) / np.sum(ys_ec)

fig, ax = plt.subplots(1, 1)
# Reward distribution
plt.plot(ec.x, ec.p_prior, color="#999999")
# Efficient code density
plt.plot(ec.x, ys_ec, color="#2D59B3")
# kde estimate of real neurons
plt.plot(ec.x, ys, color="k")
# individual neurons
ax.vlines(true_thresh, 0, 0.012 * np.ones(true_thresh.shape), colors="k")
# Decoration
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(0, 25)
ax.set_ylim(0, 0.22)
ax.set_xticks([0, 10, 20])
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
fig.set_figwidth(6.2)
fig.set_figheight(2)
plt.legend(
    ["reward density", "efficient code", "measured neurons"], fontsize=14, frameon=False
)
plt.xlabel("Thresholds", fontdict={"size": 16})

plt.savefig(os.path.join(".", fig_folder, "thresholds.png"), bbox_inches="tight")
plt.savefig(os.path.join(".", fig_folder, "thresholds.pdf"), bbox_inches="tight")
# # Threshold - Gain plot
print("Correlation result for thresholds and gains")
print(scipy.stats.pearsonr(true_thresh, param_set[:, 1]))

pars_ec = ec.get_sigmoid_fits()
gain_ec = pars_ec[:, 1]

fig4, ax4 = plt.subplots(1, 1)
ax4.set_xticks([0, 5, 10, 20])
ax4.scatter(true_thresh, param_set[:, 1], s=10, color=[0, 0, 0])
ax4.plot(thresh_ec[:-1], gain_ec[:-1], ".-", color="#2D59B3")
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 50)
fig4.set_figwidth(4.5)
fig4.set_figheight(4.5)
ax4.spines["right"].set_visible(False)
ax4.spines["top"].set_visible(False)
ax4.set_xlabel("Threshold")
ax4.set_ylabel("Gain")

fig4.savefig(
    os.path.join(".", fig_folder, "threshold-gain-fit.png"), bbox_inches="tight"
)
fig4.savefig(
    os.path.join(".", fig_folder, "threshold-gain-fit.pdf"), bbox_inches="tight"
)

# # Threshold - Asymmetry plot

fig, ax = plt.subplots(1, 1)
RPs = ec.get_quantiles_RPs(quantiles_ec)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.scatter(true_thresh[idx_sorted], asymM_all, s=10, color="k")
ax.plot(thresh_ec, quantiles_ec, ".-", color="#2D59B3")
ax.set_xticks([0, 5, 10, 20])
ax.set_xlim([0, 12])
ax.set_ylim([0, 1])
plt.grid(False)
fig.set_figwidth(4.5)
fig.set_figheight(4.5)

RPSS = np.empty(1000)
for i, q in enumerate(np.linspace(0, 1, 1000)):
    RPSS[i] = scipy.stats.expectile(juice_magnitudes, alpha=q, weights=juice_prob)
# ax.plot(np.linspace(0,1,1000), RPSS, '--', color=[.7,.7,.7])
ax.plot(RPSS, np.linspace(0, 1, 1000), "--k")

plt.xlabel("Threshold")
plt.ylabel("Asymmetry")

fig.savefig(os.path.join(".", fig_folder, "threshold-asymm.png"), bbox_inches="tight")
fig.savefig(os.path.join(".", fig_folder, "threshold-asymm.pdf"), bbox_inches="tight")

print("Correlation for threshold and asymmetry:")
print(pearsonr(true_thresh[idx_sorted], asymM_all))

# scatter plot between thresholds/midpoints and slopes
fig, ax = plt.subplots(1, 2)

use_thresholds = True

pars_ec = ec.get_sigmoid_fits()
if use_thresholds:
    thresh_dat = true_thresh
    thresh_ec_local = np.array(thresh_ec)
else:
    thresh_dat = param_set[:, 2]
    thresh_ec_local = pars_ec[:, 2]
slope_dat = param_set[:, 0]
slope_ec = pars_ec[:, 0]

# Plot threshold slope
ax[0].scatter(thresh_dat, slope_dat, s=10, color="k")
ax[1].plot(thresh_ec_local, slope_ec, ".-", color="#2D59B3")
fig.set_figwidth(7)
fig.set_figheight(3)


func = lambda x, slope, intercept: slope*(x) + intercept
num_sim = int(5e3)
ps = []
for i in range(num_sim):
    i_sample = np.random.choice(np.arange(len(thresh_dat)-1,dtype=int),len(thresh_dat))
    while True:
        try:
            # fit the line for _thresholds and _slopes
            slope, intercept, r_value, p_value, std_err = stats.linregress(thresh_dat[i_sample], slope_dat[i_sample])
        except:
            print('redo it with different random seed.')
        break

    ps.append([slope, intercept])

slope, intercept, r_value, p_value, std_err = stats.linregress(thresh_dat, slope_dat)
ax[0].plot([0, 12], [intercept, intercept + slope * 12], 'k', lw=2)
# bow tie shaped shaded plot for the 90% confidence interval
ysample = np.asarray([func(np.linspace(0, 12, 100), *pi) for pi in ps])
lower = np.percentile(ysample, 5, axis=0)
upper = np.percentile(ysample, 95, axis=0)
ax[0].fill_between(np.linspace(0, 12, 100), lower, upper,
                   color='black', alpha=0.15)
# Decoration
for i in range(2):
    if use_thresholds:
        ax[i].set_xlabel("Threshold")
    else:
        ax[i].set_xlabel("Midpoint")
    ax[i].set_ylabel("Slope")
    ax[i].spines["right"].set_visible(False)
    ax[i].spines["top"].set_visible(False)
    ax[i].set_xlim([0, 12])
    if i == 0:
        ax[i].set_ylim([0, 0.5])
    else:
        ax[i].set_ylim([0, 2.5])

if use_thresholds:
    fig.savefig(
        os.path.join(".", fig_folder, "threshold-slope_model_sbs_diffaxis.png"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(".", fig_folder, "threshold-slope_model_sbs_diffaxis.pdf"),
        bbox_inches="tight",
    )
else:
    fig.savefig(
        os.path.join(".", fig_folder, "midpoint-slope_model_sbs_diffaxis.png"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(".", fig_folder, "midpoint-slope_model_sbs_diffaxis.png"),
        bbox_inches="tight",
    )

# correlation analysis for r and p
print("Correlation results for threshold and slope (data):")
print(pearsonr(thresh_dat, slope_dat))
print("Correlation results for threshold and slope (model):")
print(pearsonr(thresh_ec_local, slope_ec))

# # Illustration plots comparing efficient code and raw data
plt.figure(figsize=[5, 2.5])
dat = sio.loadmat("measured_neurons/dat_eachneuron_bc.mat")
dat = dat["dat"]
juiceAmounts = [0.1, 0.3, 1.2, 2.5, 5, 10, 20]
y = np.array([np.nanmean(dat[0, i][0], 0) for i in range(dat.shape[1])]).T
plt.plot(juiceAmounts, y - np.min(y, 0, keepdims=True), "k-", alpha=0.5)
plt.ylim(bottom=0, top=30)
plt.xlim(left=0, right=20)
plt.xticks(juiceAmounts, labels=[])
plt.yticks([0, 10, 20, 30], labels=[])
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.savefig("figures/raw_data.pdf")

plt.figure(figsize=[5, 2.5])
mu, sigma = get_pars()
x = np.linspace(np.finfo(float).eps, 100, 1000)
y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
plot_optimal(
    plt.gca(),
    x,
    y_pdf,
    np.cumsum(y_pdf) / (0.0001 + np.sum(y_pdf)),
    alpha,
    20,
    mu,
    sigma,
    slope_scale=slope_scale,
    color="#2D59B3",
    R_max=R_t,
)
plt.ylim(bottom=0, top=30)
plt.xlim(left=0, right=20)
plt.xticks([0, 5, 10, 15, 20], labels=[])
plt.yticks([0, 10, 20, 30], labels=[])
plt.savefig("figures/raw_model.pdf")

# illustration plot, solution for different alpha
make_solution(slope_scale=slope_scale)
plt.savefig("figures/solution_alphas.pdf")

# illustration plot, solution is efficient
make_efficient(alpha=alpha, slope_scale=slope_scale, use_beta=True)
plt.savefig("figures/efficiency_comparison.pdf")

print("mean of reward distribution")
print(np.sum(juice_magnitudes * juice_prob))
print("mean data threshold")
print(np.mean(true_thresh))
print("mean model threshold")
print(np.mean(thresh_ec))
print("mean data midpoint")
print(np.mean(midpoints))
print("mean model midpoint")
print(np.mean(pars_ec[:, 2]))
print("mean data zero-crossing estimated:")
print(np.mean(zero_crossings_estimated))

print("t-test midpoints against mean reward:")
print(scipy.stats.ttest_1samp(midpoints, np.sum(juice_magnitudes * juice_prob)))
print("t-test thresholds against mean reward:")
print(scipy.stats.ttest_1samp(true_thresh, np.sum(juice_magnitudes * juice_prob)))
