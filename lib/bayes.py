"""Posterior, priors, proposals, MH sampler and likelihood"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import cho_factor, cho_solve

from lib.cosmology import FLRW
from lib.dataset import SNDataset

class Posterior:
    def __init__(
        self,
        log_likelihood: Callable[[np.ndarray], float],
        log_prior: Callable[[np.ndarray], float],
    ):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior

    def log_prob(self, theta: np.ndarray) -> float:
        """Returns log_prior(theta) + log_likelihood(theta).
        If the prior is -inf (theta rejected), returns -inf
        """
        lp = self.log_prior(theta)
        if lp == -np.inf:
            return -np.inf
        return lp + self.log_likelihood(theta)

    def __call__(self, theta: np.ndarray) -> float:
        return self.log_prob(theta)

def flat_log_prior(bounds: Sequence[Tuple[float, float]]):
    """0 inside bounds, -inf outside. 
    Input: sequence of (lo, hi) pairs, one per parameter.
    """
    return lambda theta: 0.0 if all(lo <= t <= hi for t, (lo, hi) in zip(theta, bounds)) else -np.inf

class Likelihood:
    """Gaussian likelihood for the parameters theta = (omega_m, omega_lambda).
    Observed distance modulus is mu_obs = mu_th(z; Omega_m, Omega_lambda; H0) + M,
    where M is marginalized analytically, assuming a flat prior (-inf,inf).
    """

    def __init__(self, data: SNDataset, H0: float = 70.0):
        self.data = data
        self.H0 = H0
        self._chol = cho_factor(data.cov) #Cholesky decomposition of the covariance matrix
        ones = np.ones(len(data)) #vector of ones to compute C^-1 1
        self._Cinv_ones = cho_solve(self._chol, ones) #C^-1 1
        self._C_sum = ones @ self._Cinv_ones #1^T C^-1 1

    def distance_modulus_model(self, omega_m: float, omega_lambda: float) -> np.ndarray:
        model = FLRW(self.H0, omega_m, omega_lambda)
        return model.mu(self.data.z) #function mu calculated for z vals from dataset

    def chi2(self, omega_m: float, omega_lambda: float) -> float:
        """Chi-square, minimised analytically over the zero-point M.
        chi^2 = (d - M 1)^T C^-1 (d - M 1) = (d^T C^-1 d) - 2M (d^T C^-1 1) + M^2 (1^T C^-1 1)
        A, B, D = (d^T C^-1 d), (d^T C^-1 1), (1^T C^-1 1)
        L = exp(-0.5 * chi2) = exp(-0.5 * (A - 2MB + M^2 D))
        Integrating over M with a flat prior: 
        L = int(exp(-0.5 * (A - 2MB + M^2 D)) dM) = exp(-0.5*(A - B^2/D))int(exp(-0.5*D*(M - B/D)^2) dM),
        int(exp(-0.5*D*(M - B/D)^2) dM) = sqrt(2*pi/D) -- const factor ignored to compute log-likelihood
        """
        mu_th = self.distance_modulus_model(omega_m, omega_lambda)
        if not np.all(np.isfinite(mu_th)):
            return np.inf

        delta0 = self.data.mu - mu_th #d
        Cinv_delta0 = cho_solve(self._chol, delta0) #C^-1 d
        A = delta0 @ Cinv_delta0 #d^T C^-1 d
        B = delta0 @ self._Cinv_ones #d^T C^-1 1
        return A - B**2/self._C_sum #A - B^2/D

    def __call__(self, theta: np.ndarray) -> float:
        """Log-likelihood for the parameter vector theta"""
        omega_m, omega_lambda = theta
        return -0.5*self.chi2(omega_m, omega_lambda)

class Proposal(ABC):
    """proposal distribution q(theta_to | theta_from)."""

    @abstractmethod
    def propose(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a new point given the current one."""

    @abstractmethod
    def logpdf(self, theta_to: np.ndarray, theta_from: np.ndarray) -> float:
        """log q(theta_to | theta_from)."""

class GaussianRandomWalk(Proposal):
    """theta' = theta + N(0, dt*cov),
    where dt is the step size and cov is the covariance matrix of the proposal distribution
    """

    def __init__(self, cov: np.ndarray, dt: float = 1.0):
        self.dt = dt
        self.cov = dt * np.atleast_2d(np.asarray(cov, dtype=float))
        self.n_dim = self.cov.shape[0] 
        self._L = np.linalg.cholesky(self.cov) # L: lower-triangular sqrt of cov, used to draw samples (L L^T = cov)
        self._chol = cho_factor(self.cov) # Cholesky factor of the covariance matrix
        self._log_det_cov = 2.0*np.sum(np.log(np.diag(self._chol[0])))

    def propose(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal(self.n_dim) #normal vector (random numbers drawn from curve w/ mean 0 and var 1)
        return theta + self._L @ z #proposed new point

    def logpdf(self, theta_to: np.ndarray, theta_from: np.ndarray) -> float:
        """Log-probability of landing on `theta_to` given `theta_from` under a Gaussian random walk.
        log N(theta_to; mean=theta_from, cov=self.cov)
        log q(theta_to | theta_from) = -(ndim/2)*log(2pi) - (1/2)*log(|cov|) - (1/2)*(theta_to-theta_from)^T cov^-1 (theta_to-theta_from)
        log(|cov|) = self._log_det_cov, cov^-1 (theta_to-theta_from) = cho_solve(self._chol, theta_to-theta_from)
        """
        diff = theta_to - theta_from
        cov_inv_diff = cho_solve(self._chol, diff)
        return -(self.n_dim/2)*np.log(2*np.pi) - 0.5*self._log_det_cov - 0.5*np.dot(diff, cov_inv_diff)

@dataclass
class SamplerResult:
    """Raw output of a chain: samples, their log-posterior, and acceptance flags."""

    samples: np.ndarray  # (n_steps + 1, n_dim), includes the initial point
    log_prob: np.ndarray  # (n_steps + 1,)
    accepted: np.ndarray  # (n_steps,) bool, one flag per proposed move

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted)) #calculates proportion of True values in the accepted array

class MetropolisHastings:

    def __init__(self, posterior: Posterior, proposal: Proposal):
        self.posterior = posterior
        self.proposal = proposal

    def sample(
        self,
        theta0: np.ndarray,
        n_steps: int,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> SamplerResult:
        """Runs chain for n_steps starting at theta0"""
        rng = np.random.default_rng(rng)
        n_dim = len(theta0)
        samples = np.empty((n_steps + 1, n_dim))
        log_prob = np.empty(n_steps + 1)
        accepted = np.empty(n_steps, dtype=bool)

        samples[0] = theta0
        log_prob[0] = self.posterior.log_prob(theta0)

        for i in range(1, n_steps+1):
            theta_current = samples[i - 1]
            log_prob_current = log_prob[i - 1]

            theta_proposed = self.proposal.propose(theta_current, rng)
            log_prob_proposed = self.posterior.log_prob(theta_proposed)

            if log_prob_current == -np.inf:
                accept = log_prob_proposed > -np.inf
            else:
                log_hastings = self.proposal.logpdf(theta_current, theta_proposed) - self.proposal.logpdf(theta_proposed, theta_current)
                log_alpha = log_prob_proposed - log_prob_current + log_hastings
                accept = np.log(rng.uniform()) < log_alpha

            if accept:
                samples[i], log_prob[i] = theta_proposed, log_prob_proposed
                accepted[i - 1] = True
            else:
                samples[i], log_prob[i] = theta_current, log_prob_current
                accepted[i - 1] = False

        return SamplerResult(samples=samples, log_prob=log_prob, accepted=accepted)

@dataclass
class Cov_tune:
    """Diagnostics for one round of pilot adaptation."""

    acceptance_rate: float
    dt: float
    cov: np.ndarray

@dataclass
class TuneProp:
    """Outcome pilot-tuning of proposal during warm-up: approximation of posterior's covariance regularized and frozen"""
    proposal: GaussianRandomWalk
    theta: np.ndarray
    rounds: list[Cov_tune]

    def tune_proposal(
        posterior: Posterior,
        theta0: np.ndarray,
        init_cov: np.ndarray,
        n_rounds: int = 5,
        n_steps_per_round: int = 2000,
        dt0: float = 1.0,
        target_acceptance: Tuple[float, float] = (0.2, 0.4),
        jitter: float = 1e-6,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> TuneProp:
        """Each round runs a short chain from where the previous left off, re-estimates the proposal's 
        shape from that round's sample covariance, regularizes with `jitter` before use in a Cholesky 
        factorization, and rescales by the Roberts-Gelman-Gilks-Rosenthal asymptotically optimal factor 2.38^2/n_dim. 
        The last round's proposal is frozen and used for the computation of the chains.
        """
        rng = np.random.default_rng(rng)
        theta = np.array(theta0, dtype=float)
        cov = np.atleast_2d(np.asarray(init_cov, dtype=float)) #initial guess for proposal covariance
        n_dim = len(theta)
        optimal_scale = 2.38**2/n_dim #Roberts-Gelman-Gilks-Rosenthal asymptotically optimal scaling
        dt = dt0 #initial step size for Gaussian random walk
        rounds = []
    
        for _ in range(n_rounds):
            proposal = GaussianRandomWalk(cov, dt=dt)
            sampler = MetropolisHastings(posterior, proposal)
            result = sampler.sample(theta, n_steps_per_round, rng=rng)
            acceptance_rate = result.acceptance_rate

            round_samples = result.samples[n_steps_per_round // 2:] #re-estimate proposal shape from this round's post-transient half
            sample_cov = np.atleast_2d(np.cov(round_samples.T)) + jitter * np.eye(n_dim) #regularize with jitter
            if np.all(np.isfinite(sample_cov)) and np.linalg.det(sample_cov) > 0:
                cov = optimal_scale*sample_cov #rescaling by optimal factor
                dt = 1.0 #if update in covariance, reset step size
            elif acceptance_rate < target_acceptance[0]: #if acceptance rate too low, decrease step size
                dt *= 0.7
            elif acceptance_rate > target_acceptance[1]: #if acceptance rate too high, increase step size
                dt *= 1.3

            theta = result.samples[-1] #last sample of this round
            rounds.append(Cov_tune(acceptance_rate=acceptance_rate, dt=dt, cov=cov)) #store round's outcome

        return TuneProp(proposal=GaussianRandomWalk(cov, dt=dt), theta=theta, rounds=rounds) #store frozen proposal with final state

