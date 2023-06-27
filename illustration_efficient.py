#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:40:34 2021

@author: heiko
"""

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import scipy
import scipy.stats
import scipy.interpolate as si
from scipy.special import gammainc, gammaincinv


def make_illustration(n_neurons=20, logscale=True, save_fig=None, alpha=1, beta=None):
    if logscale:
        x_min = 0.1
        x_max = 100
        x = np.exp(np.linspace(np.log(x_min), np.log(x_max), 1000))
    else:
        x_min = np.finfo(float).eps
        x_max = 25
        x = np.linspace(x_min, x_max, 1000)
    sig_sigmoid = 0.5 / n_neurons
    mu, sigma = get_pars()
    quant = np.linspace(
        1 / (2 * n_neurons + 1), 2 * n_neurons / (2 * n_neurons + 1), n_neurons
    )
    y_cdf = scipy.stats.norm.cdf(np.log(x), mu, sigma)
    if logscale:
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma)
    else:
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
    thresh = np.exp(scipy.stats.norm.ppf(quant, mu, sigma))

    a_sig = plt.axes([0.1, 0, 0.8, 0.55])
    for i in range(n_neurons):
        gain = 1 / (1 - scipy.stats.norm.cdf(np.log(thresh[i]), mu, sigma))
        plt.plot(
            x,
            gain
            * _sigmoid(
                scipy.stats.norm.cdf(np.log(x), mu, sigma), quant[i], sig_sigmoid
            ),
            color=((n_neurons - i) / n_neurons / 1.5) * np.ones(3),
        )
    for i in range(n_neurons):
        gain = 1 / (1 - scipy.stats.norm.cdf(np.log(thresh[i]), mu, sigma))
        plt.plot([thresh[i], thresh[i]], [gain / 2, 12], "--", color=[0.5, 0.5, 0.5])
    plt.xlim(x_min, x_max)
    plt.ylim(0, 12)
    a_sig.spines["top"].set_visible(False)
    a_sig.spines["right"].set_visible(False)
    plt.ylabel("Response")
    plt.yticks([0, 2, 4, 6, 8, 10])

    a_cdf = plt.axes([0.1, 0.55, 0.8, 0.25])
    plt.plot(x, y_cdf, "k")
    plt.ylim(0, 1)
    plt.xlim(x_min, x_max)
    for i in range(n_neurons):
        plt.plot([0, thresh[i]], [quant[i], quant[i]], "--", color=[0.5, 0.5, 0.5])
        plt.plot([thresh[i], thresh[i]], [0, quant[i]], "--", color=[0.5, 0.5, 0.5])
    a_cdf.spines["top"].set_visible(False)
    a_cdf.spines["right"].set_visible(False)
    plt.ylabel("cdf")

    a_pdf = plt.axes([0.1, 0.85, 0.8, 0.15])
    plt.plot(x, y_pdf, "k")
    plt.xlim(x_min, x_max)
    a_pdf.spines["top"].set_visible(False)
    a_pdf.spines["right"].set_visible(False)
    plt.ylabel("pdf")
    plt.ylim(ymin=0)
    if logscale:
        a_sig.set_xscale("log")
        a_cdf.set_xscale("log")
        a_pdf.set_xscale("log")
        a_sig.set_xticks([0.1, 1, 10, 100])
        a_cdf.set_xticks([0.1, 1, 10, 100])
        a_cdf.set_xticklabels([])
        a_pdf.set_xticks([0.1, 1, 10, 100])
        a_pdf.set_xticklabels([])
        a_sig.set_xlabel("Value [log]")
    else:
        a_sig.set_xticks(np.arange(0, x_max, 10))
        a_cdf.set_xticks(np.arange(0, x_max, 10))
        a_cdf.set_xticklabels([])
        a_pdf.set_xticks(np.arange(0, x_max, 10))
        a_pdf.set_xticklabels([])
        a_sig.set_xlabel("Value")
    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", bbox_inches="tight")


def make_illustration_panel(
    n_neurons=20,
    logscale=False,
    save_fig=None,
    rs=7.71,
    alpha=0.77,
    gamma=0,
    slope_scale=5,
    plot_thresh="none",
):
    if logscale:
        x_min = 0.1
        x_max = 100
        x = np.exp(np.linspace(np.log(x_min), np.log(x_max), 1000))
    else:
        x_min = np.finfo(float).eps
        x_max = 30
        x = np.linspace(x_min, x_max, 1000)
    R = 255.19 / 39 * n_neurons
    p_thresh = (2 * np.arange(n_neurons) + 1) / n_neurons / 2
    mu, sigma = get_pars()
    if logscale:
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma)
    else:
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
    y_cdf = np.cumsum(y_pdf)
    y_pdf = y_pdf / np.sum(y_pdf)
    y_cdf /= y_cdf[-1]
    d = y_pdf / (1 - scipy.stats.norm.cdf(np.log(x), mu, sigma)) ** (1 - alpha)
    cum_density = np.cumsum(d)
    cum_density /= cum_density[-1]
    thresh = np.interp(p_thresh, cum_density, x)
    quant = np.interp(thresh, x, y_cdf)
    gain_x = 1 / ((1 - scipy.stats.norm.cdf(np.log(x), mu, sigma)) ** alpha)
    spont_rate = gain_x**gamma * rs
    # a = (n_neurons + 1) * quant
    # b = (n_neurons + 1) * (1 - quant)
    a = slope_scale * quant
    b = slope_scale * (1 - quant)
    y = np.empty((n_neurons, len(x)))
    for i in range(n_neurons):
        gain = 1 / ((1 - scipy.stats.norm.cdf(np.log(thresh[i]), mu, sigma)) ** alpha)
        y[i] = gain * scipy.special.betainc(a[i], b[i], y_cdf)
        # y[i] = gain * scipy.special.expit(39 * a * (y_cdf - quant[i]))
    cost = np.sum(y * y_pdf) / np.sum(y_pdf)
    y = R * y / cost
    a_sig = plt.axes([0.1, 0, 0.8, 0.55])
    thresh_r = np.zeros_like(thresh)
    for i in range(n_neurons):
        thresh_r[i] = x[np.searchsorted(y[i], rs, side="right") - 1]
        plt.plot(x, y[i], color=((n_neurons - i) / n_neurons / 1.5) * np.ones(3))
    if plot_thresh == "mid":
        for i in range(n_neurons):
            mid = np.interp(thresh[i], x, y[i])
            plt.plot([thresh[i], thresh[i]], [mid, 50], "--", color=[0.5, 0.5, 0.5])
    elif plot_thresh == "thresh":
        plt.plot(x, spont_rate, "k--")
        for i in range(n_neurons):
            plt.plot([thresh_r[i], thresh_r[i]], [rs, 50], "--", color=[0.5, 0.5, 0.5])
    else:
        plt.plot(x, spont_rate, "k--")
    plt.xlim(x_min, 25)
    plt.ylim(0, 50)
    a_sig.spines["top"].set_visible(False)
    a_sig.spines["right"].set_visible(False)
    plt.ylabel("Response")
    plt.yticks([0, 10, 20, 30, 40, 50])

    if logscale:
        a_sig.set_xscale("log")
        a_sig.set_xticks([0.1, 1, 10, 100])
        a_sig.set_xlabel("Value [log]")
    else:
        a_sig.set_xticks(np.arange(0, x_max, 10))
        a_sig.set_xlabel("Value")
    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", bbox_inches="tight")


def make_rl_illustration(logscale=False, save_fig=None, alpha=1):
    if logscale:
        x_min = 0.1
        x_max = 100
        x = np.exp(np.linspace(np.log(x_min), np.log(x_max), 1000))
    else:
        x_min = np.finfo(float).eps
        x_max = 15
        x = np.linspace(x_min, x_max, 1000)
    mu, sigma = get_pars()
    if logscale:
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma)
    else:
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x

    t0 = 5

    a_cdf = plt.axes([0.1, 0.55, 0.8, 0.25])
    plt.plot([2, 5, 7], [-0.4, 0, 1], "k-", linewidth=2)
    plt.plot([x_min, x_max], [0, 0], "k--", linewidth=1)
    # p = patches.Wedge([t0, 0], 1.5, 0, 180 / np.pi * np.arctan2(1, 2))
    p = patches.Polygon([[t0, 0], [t0 + 2, 0], [t0 + 2, 1]], facecolor=[0, 0, 1, 0.5])
    a_cdf.add_patch(p)
    p = patches.Polygon(
        [[t0, 0], [t0 - 2, 0], [t0 - 2, -0.8 / 3]], facecolor=[1, 0, 0, 0.5]
    )
    a_cdf.add_patch(p)

    a_cdf.spines["top"].set_visible(False)
    a_cdf.spines["right"].set_visible(False)
    a_cdf.spines["bottom"].set_visible(False)
    plt.xlim(x_min, x_max)
    plt.ylim(-0.5, 1.5)

    plt.ylabel("response")
    a_pdf = plt.axes([0.1, 0.85, 0.8, 0.15])
    # add blue patch
    blue_bool = x > t0
    y_p = np.concatenate([[0], y_pdf[blue_bool], [0]])
    x_p = np.concatenate([[t0], x[blue_bool], [x_max]])
    p = patches.Polygon(np.stack([x_p, y_p]).T, facecolor=[0, 0, 1, 0.5])
    a_pdf.add_patch(p)

    # add red patch
    red_bool = x <= t0
    y_p = np.concatenate([[0], y_pdf[red_bool], [0]])
    x_p = np.concatenate([[x_min], x[red_bool], [t0]])
    p = patches.Polygon(np.stack([x_p, y_p]).T, facecolor=[1, 0, 0, 0.5])
    a_pdf.add_patch(p)

    plt.plot(x, y_pdf, "k")
    plt.xlim(x_min, x_max)
    a_pdf.spines["top"].set_visible(False)
    a_pdf.spines["right"].set_visible(False)
    plt.ylabel("pdf")
    plt.ylim(ymin=0)

    n_neurons = 12
    length = 1.5
    p_thresh = (2 * np.arange(n_neurons) + 1) / n_neurons / 2
    thresh = np.exp(scipy.stats.norm.ppf(p_thresh, mu, sigma))
    asym = (1 - p_thresh) / p_thresh
    alpha_minus = np.sqrt(asym)
    alpha_plus = 1 / np.sqrt(asym)
    a_resp = plt.axes([0.1, 0.2, 0.8, 0.3])
    for i in range(n_neurons):
        angle = np.arctan2([1, 1], [alpha_plus[i], alpha_minus[i]])
        x = [
            thresh[i] - (np.cos(angle[0]) * length),
            thresh[i],
            thresh[i] + (np.cos(angle[1]) * length),
        ]
        y = [-np.sin(angle[0]) * length, 0, np.sin(angle[1]) * length]
        plt.plot(x, y, "-", color=[0, 0, 0, 0.7])
    plt.plot([x_min, x_max], [0, 0], "k--", linewidth=1)
    a_resp.spines["top"].set_visible(False)
    a_resp.spines["right"].set_visible(False)
    a_resp.spines["bottom"].set_visible(False)
    a_resp.set_xticks([])
    plt.ylabel("response")
    plt.axis("equal")

    plt.xlim(x_min, x_max)
    plt.ylim(-1.5, 1.5)
    if logscale:
        a_cdf.set_xscale("log")
        a_pdf.set_xscale("log")
        a_cdf.set_xticks([0.1, 1, 10, 100])
        a_cdf.set_xticklabels([])
        a_pdf.set_xticks([0.1, 1, 10, 100])
        a_pdf.set_xticklabels([])
        a_pdf.set_xlabel("Value [log]")
    else:
        a_cdf.set_xticks([])
        a_cdf.set_xticklabels([])
        a_pdf.set_xticks(np.arange(0, x_max + 1, 5))
        a_pdf.set_xticklabels([])
        a_pdf.set_xlabel("Value")
    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", bbox_inches="tight")


def make_rl_insets():
    # spiketrain
    noise = np.random.randn(1000)
    times = [10, 200, 400, 800, 850]
    spike = np.array(
        [
            0,
            0.25,
            0.5,
            1,
            0.8,
            0.5,
            0,
            -0.1,
            -0.2,
            -0.19,
            -0.175,
            -0.15,
            -0.125,
            -0.1,
            -0.08,
            -0.06,
            -0.04,
            -0.03,
            -0.02,
            -0.01,
        ]
    )
    response = noise
    for t in times:
        response[t : (t + len(spike))] += 10 * spike

    plt.plot(response, "k")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig("spiketrace.pdf", format="pdf", bbox_inches="tight")

    plt.figure()
    thresh = 0.2
    x = np.linspace(0, 1, 1000)
    y = scipy.special.betainc(5, 5, x)
    thresh_x = x[np.where(y > thresh)[0][0]]
    plt.plot(x, y, "k", linewidth=2)
    plt.plot([-0.1, 1], [thresh, thresh], "k--", linewidth=2, alpha=0.4)
    plt.plot([thresh_x, thresh_x], [-0.1, thresh], "k--", linewidth=2, alpha=0.4)
    plt.plot([0.5, 0.5], [-0.1, 0.5], "k", linewidth=2, alpha=0.4)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().set_xticks([thresh_x, 0.5])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticks([thresh])
    plt.gca().set_yticklabels([])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.savefig("tuning.pdf", format="pdf", bbox_inches="tight")

    # Values distribution
    plt.figure()
    mu, sigma = get_pars()
    rewards = np.array([0.1, 0.3, 1.2, 2.5, 5, 10, 20])
    ps = np.array([0.066, 0.091, 0.15, 0.15, 0.31, 0.15, 0.077])
    x = np.linspace(np.finfo(float).eps, 20, 1000)
    y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
    for r, p in zip(rewards, ps):
        plt.plot([r, r], [0, p], "k")
        # plt.plot(r, p, 'bo', fillstyle=None, markersize=10)
    plt.plot(x, 2 * y_pdf, "k", alpha=0.4, linewidth=2)
    plt.gca().set_xticks([0, 5, 10, 15, 20])
    plt.gca().set_yticks([])
    plt.gca().spines["bottom"].set_linewidth(2)
    plt.gca().tick_params(width=2)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.savefig("rewards.pdf", format="pdf", bbox_inches="tight")


def make_solution(
    n_neurons=10,
    save_fig=None,
    alphas=None,
    use_beta=True,
    mus=None,
    sigmas=None,
    slope_scale=5,
    x_max=50,
):
    if alphas is None:
        alphas = [1, 0.8, 0.6, 0.4, 0.2]
    if mus is None:
        mu0, sigma0 = get_pars()
        mus = [mu0, 1, 1.5, 2, 3]
        sigmas = [sigma0, 0.5, 1, 0.5, 0.25]
    else:
        assert len(mus) == len(
            sigmas
        ), "mu and sigma vectors must be the same length if provided"
    print("the used alpha values were:")
    print(alphas)
    print("the used mus and sigmas were:")
    print(mus)
    print(sigmas)
    plt.figure(figsize=[len(mus) * 2.5 + 1.5, (len(alphas) + 1) * 1.25 + 0.5])
    x_min = np.finfo(float).eps
    x = np.linspace(x_min, x_max, 1000)
    for j, [mu, sigma] in enumerate(zip(mus, sigmas)):
        y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
        y_pdf = y_pdf / np.sum(y_pdf)
        y_cdf = scipy.stats.norm.cdf(np.log(x), mu, sigma)
        ax_pdf = plt.subplot(len(alphas) + 1, len(mus), j + 1)
        ax_pdf.plot(x, y_pdf, "k", alpha=0.4, linewidth=2)
        ax_pdf.spines["top"].set_visible(False)
        ax_pdf.spines["right"].set_visible(False)
        ax_pdf.set_xticklabels([])
        ax_pdf.set_yticklabels([])
        for i, alpha in enumerate(alphas):
            ax = plt.subplot(len(alphas) + 1, len(mus), (i + 1) * len(mus) + j + 1)
            plot_optimal(
                ax,
                x,
                y_pdf,
                y_cdf,
                alpha,
                n_neurons,
                mu,
                sigma,
                use_beta=use_beta,
                slope_scale=slope_scale,
            )

    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", bbox_inches="tight")


def plot_optimal(
    ax,
    x,
    y_pdf,
    y_cdf,
    alpha,
    n_neurons,
    mu,
    sigma,
    use_beta=True,
    slope_scale=None,
    color="k",
    R_max=255.19
):
    if slope_scale is None:
        slope_scale = n_neurons + 1
    R = R_max / 39 * n_neurons
    p_thresh = (2 * np.arange(n_neurons) + 1) / n_neurons / 2
    y_max = 60
    resp5 = np.zeros((n_neurons, 1000))
    d = y_pdf / (1 - y_cdf) ** (1 - alpha)
    cum_density = np.cumsum(d)
    cum_density /= cum_density[-1]
    thresh = np.interp(p_thresh, cum_density, x)
    quant = np.interp(thresh, x, y_cdf)
    for i in range(n_neurons):
        gain = 1 / ((1 - scipy.stats.norm.cdf(np.log(thresh[i]), mu, sigma)) ** alpha)
        if use_beta:
            a = slope_scale * quant[i]
            b = slope_scale * (1 - quant[i])
            resp5[i] = gain * scipy.special.betainc(a, b, y_cdf)
        else:
            resp5[i] = gain * scipy.special.expit(slope_scale * (y_cdf - quant[i]))
    cost5 = np.sum(resp5 * y_pdf) / np.sum(y_pdf)
    resp5 = R * resp5 / cost5
    ax.plot(x, resp5.T, color=color)
    ax.set_xlim(0, x[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_ylabel('Response [Hz]')
    ax.set_ylim(0, y_max)
    ax.set_xticks(np.arange(0, x[-1] + 1, 10))
    ax.set_xticklabels([])
    ax.set_yticks([0, 25, 50])
    ax.set_yticklabels([])


def make_efficient(
    n_neurons=10, save_fig=None, alpha=1, slope_scale=None, use_beta=False, R_t=255.19
):
    if slope_scale is None:
        slope_scale = n_neurons + 1
    # seaborn muted first colors:
    colors = [
        (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
        (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
        (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
        (0.41568627450980394, 0.8, 0.39215686274509803),
        (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
    ]
    colors.reverse()
    eps = np.exp(-8)
    mu, sigma = get_pars()
    x_min = np.finfo(float).eps
    x_max = 30
    x = np.linspace(x_min, x_max, 1000)
    R = R_t / 39 * n_neurons
    # y_cdf = scipy.stats.norm.cdf(np.log(x), mu, sigma)
    y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
    y_cdf = np.cumsum(y_pdf)
    y_pdf = y_pdf / np.sum(y_pdf)
    y_cdf /= y_cdf[-1]
    p_thresh = (2 * np.arange(n_neurons) + 1) / n_neurons / 2

    # a single neuron
    g = get_optimal_single(x, y_pdf, r_max=15, r_mean=R)
    resp1 = si.CubicSpline(x, g)
    cost1 = np.sum(resp1(x) * y_pdf) / np.sum(y_pdf)
    resp1 = si.CubicSpline(x, R * g / cost1)
    d1 = resp1.derivative()
    info1 = d1(x) ** 2 / (resp1(x) + eps)

    # equally spaced neurons
    resp2 = np.zeros((n_neurons, 1000))
    for i in range(n_neurons):
        # resp2[i] = scipy.special.betainc(i + 1, n_neurons-i, x / x_max)
        mu_s = i * 20 / (n_neurons - 1)
        resp2[i] = scipy.special.expit(x - mu_s)
    cost2 = np.sum(resp2 * y_pdf) / np.sum(y_pdf)
    resp2 = R * resp2 / cost2
    info2 = np.zeros_like(x)
    for i in range(n_neurons):
        r = si.CubicSpline(x, resp2[i])
        d2 = r.derivative()
        info2 += d2(x) ** 2 / (r(x) + eps)

    # optimal distribution, w/o gain?
    resp3 = np.zeros((n_neurons, 1000))
    for i in range(n_neurons):
        if use_beta:
            a = slope_scale * p_thresh[i]
            b = slope_scale * (1 - p_thresh[i])
            resp3[i] = scipy.special.betainc(a, b, y_cdf)
        else:
            resp3[i] = scipy.special.expit(slope_scale * (y_cdf - p_thresh[i]))
    cost3 = np.nansum(resp3 * y_pdf) / np.sum(y_pdf)
    resp3 = R * resp3 / cost3
    info3 = np.zeros_like(x)
    for i in range(n_neurons):
        r = si.CubicSpline(x, resp3[i])
        d3 = r.derivative()
        info3 += d3(x) ** 2 / (r(x) + eps)

    # optimal distribution
    resp4 = np.zeros((n_neurons, 1000))
    thresh = np.interp(p_thresh, y_cdf, x)
    for i in range(n_neurons):
        gain = 1 / (1 - scipy.stats.norm.cdf(np.log(thresh[i]), mu, sigma))
        if use_beta:
            a = slope_scale * p_thresh[i]
            b = slope_scale * (1 - p_thresh[i])
            resp4[i] = gain * scipy.special.betainc(a, b, y_cdf)
        else:
            resp4[i] = gain * scipy.special.expit(slope_scale * (y_cdf - p_thresh[i]))
    cost4 = np.sum(resp4 * y_pdf) / np.sum(y_pdf)
    resp4 = R * resp4 / cost4
    info4 = np.zeros_like(x)
    for i in range(n_neurons):
        r = si.CubicSpline(x, resp4[i])
        d4 = r.derivative()
        info4 += d4(x) ** 2 / (r(x) + eps)

    # optimal distribution with alpha
    resp5 = np.zeros((n_neurons, 1000))
    d = y_pdf / (1 - scipy.stats.norm.cdf(np.log(x), mu, sigma)) ** (1 - alpha)
    cum_density = np.cumsum(d)
    cum_density /= cum_density[-1]
    thresh = np.interp(p_thresh, cum_density, x)
    quant = np.interp(thresh, x, y_cdf)
    for i in range(n_neurons):
        gain = 1 / ((1 - scipy.stats.norm.cdf(np.log(thresh[i]), mu, sigma)) ** alpha)
        if use_beta:
            a = slope_scale * quant[i]
            b = slope_scale * (1 - quant[i])
            resp5[i] = gain * scipy.special.betainc(a, b, y_cdf)
        else:
            resp5[i] = gain * scipy.special.expit(slope_scale * (y_cdf - quant[i]))
    cost5 = np.sum(resp5 * y_pdf) / np.sum(y_pdf)
    resp5 = R * resp5 / cost5
    info5 = np.zeros_like(x)
    for i in range(n_neurons):
        r = si.CubicSpline(x, resp5[i])
        d5 = r.derivative()
        info5 += d5(x) ** 2 / (r(x) + eps)

    y_max = 40
    y_pdf_p = 3 / 4 * y_max / np.max(y_pdf) * y_pdf

    plt.figure(figsize=(5, 14))

    a_1 = plt.axes([0.05, 0.875, 0.9, 0.12])
    plt.plot(x, y_pdf_p, "k", alpha=0.2)
    plt.plot(x, resp1(x) / n_neurons, color=colors[0])
    plt.xlim(0, 20)
    a_1.spines["top"].set_visible(False)
    a_1.spines["right"].set_visible(False)
    plt.ylabel("Response [Hz]")
    plt.ylim(0, y_max)
    a_1.set_xticks(np.arange(0, 21, 5))
    a_1.set_xticklabels([])
    a_1.set_yticks([0, 10, 20, 30])

    a_2 = plt.axes([0.05, 0.75, 0.9, 0.12])
    plt.plot(x, y_pdf_p, "k", alpha=0.2)
    plt.plot(x, resp2.T, color=colors[1])
    plt.xlim(0, 20)
    a_2.spines["top"].set_visible(False)
    a_2.spines["right"].set_visible(False)
    plt.ylabel("Response [Hz]")
    plt.ylim(0, y_max)
    a_2.set_xticks(np.arange(0, 21, 5))
    a_2.set_xticklabels([])
    a_2.set_yticks([0, 10, 20, 30])

    a_3 = plt.axes([0.05, 0.625, 0.9, 0.12])
    plt.plot(x, y_pdf_p, "k", alpha=0.2)
    plt.plot(x, resp3.T, color=colors[2])
    plt.xlim(0, 20)
    a_3.spines["top"].set_visible(False)
    a_3.spines["right"].set_visible(False)
    plt.ylabel("Response [Hz]")
    plt.ylim(0, y_max)
    a_3.set_xticks(np.arange(0, 21, 5))
    a_3.set_xticklabels([])
    a_3.set_yticks([0, 10, 20, 30])

    a_4 = plt.axes([0.05, 0.5, 0.9, 0.12])
    plt.plot(x, y_pdf_p, "k", alpha=0.2)
    plt.plot(x, resp4.T, color=colors[3])
    a_4.set_xlim(0, 20)
    a_4.spines["top"].set_visible(False)
    a_4.spines["right"].set_visible(False)
    plt.ylabel("Response [Hz]")
    plt.ylim(0, y_max)
    a_4.set_xticks(np.arange(0, 21, 5))
    a_4.set_xticklabels([])
    a_4.set_yticks([0, 10, 20, 30])

    a_5 = plt.axes([0.05, 0.375, 0.9, 0.12])
    plt.plot(x, y_pdf_p, "k", alpha=0.2)
    plt.plot(x, resp5.T, color=colors[4])
    plt.xlim(0, 20)
    a_5.spines["top"].set_visible(False)
    a_5.spines["right"].set_visible(False)
    plt.ylabel("Response [Hz]")
    plt.ylim(0, y_max)
    a_5.set_xticks(np.arange(0, 21, 5))
    a_5.set_xticklabels([])
    a_5.set_yticks([0, 10, 20, 30])

    a_info = plt.axes([0.05, 0.155, 0.9, 0.2])
    plt.plot(x, np.log2(info1), color=colors[0])
    plt.plot(x, np.log2(info2), color=colors[1])
    plt.plot(x, np.log2(info3), color=colors[2])
    plt.plot(x, np.log2(info4), color=colors[3])
    plt.plot(x, np.log2(info5), color=colors[4])
    plt.xlim(0, 20)
    a_info.spines["top"].set_visible(False)
    a_info.spines["right"].set_visible(False)
    plt.ylabel("Fisher information [log]")
    plt.ylim(0, 10)
    a_info.set_xticks(np.arange(0, 21, 5))

    a_bar = plt.axes([0.05, 0.025, 0.9, 0.1])
    expected_info = [
        np.sum(np.log2(info1) * y_pdf),
        np.sum(np.log2(info2) * y_pdf),
        np.sum(np.log2(info3) * y_pdf),
        np.sum(np.log2(info4) * y_pdf),
        np.sum(np.log2(info5) * y_pdf),
    ]
    plt.bar(np.arange(5), expected_info - expected_info[0], color=colors)
    plt.plot([-0.5, 4.5], [0, 0], "k--")
    plt.ylabel("$\Delta E(\log I_f)$", usetex=True)
    a_bar.spines["top"].set_visible(False)
    a_bar.spines["right"].set_visible(False)
    a_bar.set_xticks(np.arange(5))
    a_bar.set_xticklabels(
        ["one neuron", "equal spacing", "no gain", "optimal", "optimal_alpha"]
    )

    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", bbox_inches="tight")


def integrate_single(x, p, lambda_r=1, f_start=1e-5, d_start=1e-5, eps=1e-7):
    """solves the differential equation for the single cell solution
    via simple iteration.
    the equation was:
        f'(x)/ f(x) = 2 p'(x)/p(x) + f(x)/F(x) + lambda_r f(x)
    """
    f = f_start
    d = d_start
    p = si.CubicSpline(x, p)
    p_d = p.derivative()
    x_last = x
    f_out = np.zeros_like(x)
    for i, x_i in enumerate(x):
        d2 = d * (p_d(x_i) / (p(x_i) + eps) + d / (f + eps) + lambda_r * d)
        d += d2 * (x - x_last)
        f += d * (x - x_last)
        x_last = x_i
        f_out[i] = f
    return f


def get_optimal_single(x, y_pdf, alpha=1.0, r_max=1.0, r_mean=1.0, a_thresh=10**-6):
    q = 1 - alpha / 2
    y_pdf /= np.sum(y_pdf)  # just to be sure
    u = np.cumsum(y_pdf)
    u = u / u[-1]
    g0 = r_max * u ** (1 / q)
    cost0 = np.sum(y_pdf * g0)
    if cost0 < r_mean:
        return g0
    a1 = 1
    a0 = 0
    g = r_max * (gammaincinv(q, u * gammainc(q, a1)) / a1) ** alpha
    cost1 = np.sum(y_pdf * g)
    while cost1 > r_mean:
        a1 = 2 * a1
        g = r_max * (gammaincinv(q, u * gammainc(q, a1)) / a1) ** alpha
        cost0 = cost1
        cost1 = np.sum(y_pdf * g)
    # now the best value for a is between a0 and a1
    while a1 - a0 > a_thresh:
        a = a0 + (a1 - a0) / 2
        g = r_max * (gammaincinv(q, u * gammainc(q, a1)) / a1) ** alpha
        cost = np.sum(y_pdf * g)
        if cost < r_mean:
            cost1 = cost
            a1 = a
        else:
            cost0 = cost
            a0 = a
    return g


def _sigmoid(x, mu, sigma):
    return scipy.special.expit((x - mu) / sigma)


def _sigmoid_d(x, mu, sigma):
    x_n = (x - mu) / sigma
    return np.exp(x_n) / ((1 + np.exp(x_n)) ** 2) / sigma


def plot_sum_d(n=5):
    # x_t = np.linspace(-1 / 10, 1 + 1 / 10, n)
    x_t = np.linspace(0, 1, n)
    sigma = 0.5 / n
    x = np.linspace(0, 1, 1000)
    y = np.zeros(1000)
    for i in range(n):
        y += _sigmoid_d(x, x_t[i], sigma)
    plt.plot(x, y)
    plt.ylim(ymin=0)


def get_pars():
    rewards = np.array([0.1, 0.3, 1.2, 2.5, 5, 10, 20])
    ps = np.array([0.066, 0.091, 0.15, 0.15, 0.31, 0.15, 0.077])
    # fitted parameters
    # mu = np.sum(np.log(rewards) * ps) / np.sum(ps)
    # v = np.sum((np.log(rewards)-mu_ml)**2 * ps) / np.sum(ps)
    # sigma = np.sqrt(v_ml)
    # ML from exact presentation numbers:
    # mu = 0.9802054796
    # sigma = 1.41029125
    # moment matching parameters
    mean = np.sum(rewards * ps) / np.sum(ps)
    var = np.sum((rewards - mean) ** 2 * ps) / np.sum(ps)
    sigma = np.log(var / mean**2 + 1)
    mu = np.log(mean) - (sigma / 2)
    sigma = np.sqrt(sigma)
    # mu = 1.297
    # sigma = 0.841
    # close nice interpretation parameters:
    # mu = 1
    # sigma = np.sqrt(2)
    return mu, sigma


# make_illustration(n_neurons=10, logscale=False, save_fig='illustration_1.pdf')
# plt.figure()
# make_rl_illustration(logscale=False, save_fig='illustration_2.pdf')
# make_efficient()
# x_min = np.finfo(float).eps
# x_max = 25
# x = np.linspace(x_min, x_max, 1000)
# mu, sigma = get_pars()
# y_pdf = scipy.stats.norm.pdf(np.log(x), mu, sigma) / x
# g = get_optimal_single(x, y_pdf, r_mean=0.1)
# make_illustration_panel()
