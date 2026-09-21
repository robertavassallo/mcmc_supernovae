"""Diagnostics for MCMC chains: 
autocorrelation, ESS, split-Rhat, power spectrum, Dunkley fit, convergence report and plots
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalised autocorrelation rho_j = Cov(x_t, x_{t+j})/Var(x)"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    delta_x = x - x.mean()
    var = np.dot(delta_x, delta_x)/n
    if var == 0.0:
        return np.zeros(max_lag + 1)

    #find size to enlarge array adding zeros to prevent FFT from introducing 'circular' correlations
    #FFT is optimized to work with sizes that are powers of 2
    size = 1
    while size < 2*n:
        size *= 2
    fx = np.fft.rfft(delta_x, n=size) #FFT
    acf = np.fft.irfft(fx*np.conjugate(fx))[:n].real #inverse FFT of squared signal (P(k)) to get autocorrelation function
    acf /= n*var #normalization
    return acf[: max_lag + 1]

def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time tau_int:
    how many steps takes to produce one independent set of parameters"""    
    rho = autocorrelation(x)
    tau = 1.0
    for m in range(1, len(rho)):
        tau = 1.0 + 2.0*np.sum(rho[1 : m + 1])
        if m >= c*tau: #Sokal's automated windowing criterion
            return max(tau, 1e-6)
    return max(tau, 1e-6)

def effective_sample_size(x: np.ndarray) -> float:
    """N_eff = N/tau_int"""
    n = len(x)
    tau = integrated_autocorr_time(x)
    return n/tau

def split_rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin split-Rhat:
    ratio between global var of all chains and local var of individual chains"""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    n_chains, n_steps = chains.shape
    split = n_steps//2
    #split each chain in half
    split_chains =  chains[:, :2*split].reshape(n_chains*2, split) 
    m, n = split_chains.shape

    chain_means = split_chains.mean(axis=1)
    chain_vars = split_chains.var(axis=1, ddof=1)

    #areas that split-chains explore -- average variance of the chains
    w = np.mean(chain_vars)
    if w == 0:
        return 1.0
    b = n*chain_means.var(ddof=1) #variance between chains
    v = ((n - 1)/n)*w + b/n #estimated total variance 
    return np.sqrt(v/w) #split-Rhat

@dataclass
class ConvergenceReport:
    rhat: Dict[str, float]
    ess: Dict[str, float]
    tau_int: Dict[str, float]
    acceptance_rate: float
    passed: Dict[str, bool]
    ok: bool

def convergence_report(
    chains: Dict[str, np.ndarray],
    accepted: np.ndarray,
    rhat_threshold: float = 1.01,
    ess_min: float = 200.0,
    acceptance_bounds: Tuple[float, float] = (0.15, 0.5),
) -> ConvergenceReport:
    """Used to decide if run converged and can be used"""
    rhat = {}
    ess = {}
    tau_int = {}
    passed = {}

    for name, chain in chains.items():
        chain_2d = np.atleast_2d(np.asarray(chain, dtype=float))
        rhat[name] = split_rhat(chain_2d)
        tau_int[name] = float(np.mean([integrated_autocorr_time(c) for c in chain_2d]))
        ess[name] = float(np.sum([effective_sample_size(c) for c in chain_2d]))
        passed[name] = (rhat[name] < rhat_threshold) and (ess[name] >= ess_min)

    acceptance_rate = np.mean(accepted)
    #check if all parameters converged and if acceptance rate is within bounds 
    #(too high: step size too small, too low: step size too large)
    ok = all(passed.values()) and (acceptance_bounds[0] <= acceptance_rate <= acceptance_bounds[1])

    return ConvergenceReport(
        rhat=rhat,
        ess=ess,
        tau_int=tau_int,
        acceptance_rate=acceptance_rate,
        passed=passed,
        ok=ok
    )

def power_spectrum(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Power spectrum of a post-warm-up chain.
    a_j = (1/sqrt(N)) sum_n(x_n - x_bar)exp(2*pi*i*j*n/N), P_j = |a_j|^2,
    k_j = 2*pi*j/N"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    delta_x = x - x.mean() #x - x_bar

    fx = np.fft.rfft(delta_x) #sqrt(N)*a_j, for j = 0, 1, ..., N//2
    P = (np.abs(fx)**2)/n #|a_j|^2
    j = np.arange(len(fx))
    k = 2.0*np.pi*j/n
    return k[1:], P[1:]

@dataclass
class Dunkley_diagnostics:
    """Dunkley et al. power-spectrum diagnostic: fit of
    P(k) = P0/(1 + (k/k*)^alpha)"""
    P0: float
    k_star: float
    j_star: float
    alpha: float
    r: float
    passed: bool

def log_binned_trend(
    k: np.ndarray, P: np.ndarray, min_per_bin: int = 8, growth: float = 1.3
) -> Tuple[np.ndarray, np.ndarray]:
    """Average P_j in log spaced k bins so the trend of P(k) can be fitted"""
    n = len(k) #number of k modes
    edges = [0] #k indices of the bin edges
    width = float(min_per_bin) #minimum number of points per bin
    while edges[-1] < n:
        edges.append(min(edges[-1] + max(min_per_bin, round(width)), n)) #add next bin edge
        width *= growth #log growth of bin width

    k_binned, P_binned = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k_binned.append(k[lo:hi].mean()) #average k in bin
        P_binned.append(P[lo:hi].mean()) #average P(k) in bin
    return np.array(k_binned), np.array(P_binned)

def fit_power_spectrum(
    x: np.ndarray,
    j_star_min: float = 20.0,
    r_max: float = 0.01,
) -> Dunkley_diagnostics:
    """Fits Dunkley power-spectrum model to the log-binned trend of the P(k) of a chain 
    and returns (P0, j_star, alpha, r).
    j_star > j_star_min: enough low-frequency modes resolve the plateau
    r < r_max: mean estimate is not dominated by low-frequency modes
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    k, P = power_spectrum(x)
    k_trend, P_trend = log_binned_trend(k, P)

    log_model = lambda k, log_P0, log_k_star, alpha: log_P0 - np.log1p((k/np.exp(log_k_star))**alpha) #Dunkley power-spectrum model in log-log space
    #initial guess
    P0_guess = P_trend[:3].mean() #P0 is the average of first 3 low k bins
    below_half = np.convolve(P_trend < 0.5*P0_guess, np.ones(3), mode="valid") == 3 #3 consecutive bins where P_trend drops below 50% of P0_guess
    runs = np.flatnonzero(below_half)
    k_star_guess = k_trend[runs[0]] if runs.size else k_trend[-1] #k where P_trend first drops below 50% of P0_guess or last k
    p0 = [np.log(P0_guess), np.log(k_star_guess), 2.0] #log(P0), log(k_star), alpha

    bounds = ([np.log(P_trend.min()) - 5.0, np.log(k_trend.min()), 0.0], 
              [np.log(P_trend.max()) + 5.0, np.log(k_trend.max()), 50.0],) #keeping P0/k_star tied to the range probed by the data and alpha >= 0
    (log_P0, log_k_star, alpha), _ = curve_fit(
        log_model, k_trend, np.log(P_trend), p0=p0, bounds=bounds, maxfev=10000
    ) #best fit parameters for the Dunkley power-spectrum model on log-binned trend of P(k)
    P0 = np.exp(log_P0)
    k_star = np.exp(log_k_star)
    j_star = k_star*n/(2.0*np.pi)

    s2_x = x.var(ddof=1) #marginal chain variance
    r = P0/(n*s2_x) #Var(x_bar)/s_x^2, where Var(x_bar) ~= P0/N
    passed = (j_star > j_star_min) and (r < r_max)

    return Dunkley_diagnostics(P0=P0, k_star=k_star, j_star=j_star, alpha=alpha, r=r, passed=passed)

def trace_plot(samples: np.ndarray, param_names: Optional[Sequence[str]] = None, burnin: int = 0):
    samples = np.atleast_2d(samples)
    n_params = samples.shape[1]
    param_names = list(param_names) if param_names else [f"theta_{i}" for i in range(n_params)]

    fig, axes = plt.subplots(n_params, 1, sharex=True, figsize=(7, 2.2 * n_params), squeeze=False)
    axes = axes[:, 0]
    for i, ax in enumerate(axes):
        ax.plot(samples[:, i], lw=0.6)
        if burnin:
            ax.axvline(burnin, color="k", ls="--", lw=0.8)
        ax.set_ylabel(param_names[i])
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    return fig

def acceptance_plot(accepted: np.ndarray, window: int = 200, x_lim = None):
    accepted = np.asarray(accepted, dtype=float)
    window = min(window, len(accepted))
    running = np.convolve(accepted, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(running)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.axhline(accepted.mean(), color="k", ls="--", label=f"overall = {accepted.mean():.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel(f"acceptance rate ({window}-step window)")
    ax.legend()
    fig.tight_layout()
    return fig
