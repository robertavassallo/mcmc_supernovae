"""MCMC analysis for the Union2.1 supernova dataset.

usage: python run_mcmc.py --prior_range Omegam_min Omegam_max Omegal_min Omegal_max --nsteps N --nchains M --burnin B [--save_path PATH]"""

from __future__ import annotations
import argparse as ap
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.dataset import load_union21
from lib.bayes import MetropolisHastings, Posterior, Likelihood, flat_log_prior, TuneProp
from lib.diagnostics import convergence_report


def initialize_chains(nchains, prior_range, posterior, max_resample=50):
   """Initializes the chains chosing dispersed starting points via Latin Hypercube sampling in order to cover entire prior space"""

   om_span, ol_span = prior_range[1] - prior_range[0], prior_range[3] - prior_range[2]
   margin = 0.1  #inset from edges
   lo = np.array([prior_range[0] + margin*om_span, prior_range[2] + margin*ol_span])
   hi = np.array([prior_range[1] - margin*om_span, prior_range[3] - margin*ol_span])

   lhs = qmc.LatinHypercube(d=2, seed=0)
   starts = qmc.scale(lhs.random(n=nchains), lo, hi)

   #test starting points and discard non-finite log_prob candidates 
   #e.g. cosmologies where expansion history cannot be computed back to the data's redshifts
   rng = np.random.default_rng(0)
   n_resampled = 0
   for i, candidate in enumerate(starts):
      attempts = 0
      while not np.isfinite(posterior.log_prob(candidate)) and attempts < max_resample:
         candidate = rng.uniform(lo, hi)
         attempts += 1
         n_resampled += 1
      if not np.isfinite(posterior.log_prob(candidate)):
         raise RuntimeError(f"could not find a valid start for chain {i} after {max_resample} resamples -- narrow prior_range")
      starts[i] = candidate

   if n_resampled:
      print(f"resampled {n_resampled} candidate starting point(s) that landed on non-finite log_prob (unphysical cosmologies)")
   return starts

def plot_contours(om_chains, ol_chains, save_path=None):
   #posterior estimates: median with asymmetric 68% (16th/84th percentile) errors
   om_samples, ol_samples = om_chains.ravel(), ol_chains.ravel()

   om_lo, om_med, om_hi = np.percentile(om_samples, [16, 50, 84])
   ol_lo, ol_med, ol_hi = np.percentile(ol_samples, [16, 50, 84])
   print(f"Omega_m      = {om_med:.3f} +{om_hi - om_med:.3f} -{om_med - om_lo:.3f}")
   print(f"Omega_lambda = {ol_med:.3f} +{ol_hi - ol_med:.3f} -{ol_med - ol_lo:.3f}")

   #joint histogram + HPD levels enclosing 68%/95% of the posterior mass
   H, om_edges, ol_edges = np.histogram2d(om_samples, ol_samples, bins=60)
   H = H.T  #(X=Om, Y=OL)
   om_centers, ol_centers = 0.5*(om_edges[:-1] + om_edges[1:]), 0.5*(ol_edges[:-1] + ol_edges[1:]) 

   h_sorted = np.sort(H.ravel())[::-1]
   cum = np.cumsum(h_sorted)/h_sorted.sum()
   level_68 = h_sorted[np.searchsorted(cum, 0.68)]
   level_95 = h_sorted[np.searchsorted(cum, 0.95)]

   #corner plot: Om/OL marginals (with median + 68% band) around the 2D joint contour
   fig, axes = plt.subplots(
      2, 2, figsize=(7, 7),
      gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3], "hspace": 0.05, "wspace": 0.05},
   )
   ax_om, ax_blank, ax_joint, ax_ol = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
   ax_blank.axis("off")

   #Om marginal (top-left)
   ax_om.hist(om_samples, bins=60, color="#3182bd", alpha=0.7)
   for v, ls in [(om_med, "-"), (om_lo, "--"), (om_hi, "--")]:
      ax_om.axvline(v, color="k", lw=0.8, ls=ls)
   ax_om.set_xlim(om_edges[0], om_edges[-1])
   ax_om.set_yticks([])
   ax_om.tick_params(labelbottom=False)

   #joint 2D panel (bottom-left)
   ax_joint.contourf(om_centers, ol_centers, H, levels=[level_95, level_68, H.max()],
                     colors=["#c6dbef", "#3182bd"])
   ax_joint.contour(om_centers, ol_centers, H, levels=[level_95, level_68], colors="k", linewidths=0.8)
   ax_joint.scatter(om_med, ol_med, marker="+", color="k", s=80)
   om_line = np.linspace(0, 1.0, 100)
   ax_joint.plot(om_line, 1 - om_line, "k--", lw=0.8, label=r"flat universe ($\Omega_k=0$)")
   ax_joint.set_xlim(om_edges[0], om_edges[-1])
   ax_joint.set_ylim(ol_edges[0], ol_edges[-1])
   ax_joint.set_xlabel(r"$\Omega_m$")
   ax_joint.set_ylabel(r"$\Omega_\Lambda$")
   ax_joint.legend(loc="upper left", fontsize=8)

   #OL marginal (bottom-right, rotated to share the joint panel's y-axis)
   ax_ol.hist(ol_samples, bins=60, orientation="horizontal", color="#3182bd", alpha=0.7)
   for v, ls in [(ol_med, "-"), (ol_lo, "--"), (ol_hi, "--")]:
      ax_ol.axhline(v, color="k", lw=0.8, ls=ls)
   ax_ol.set_ylim(ol_edges[0], ol_edges[-1])
   ax_ol.set_xticks([])
   ax_ol.tick_params(labelleft=False)

   fig.suptitle(
      rf"$\Omega_m={om_med:.3f}^{{+{om_hi - om_med:.3f}}}_{{-{om_med - om_lo:.3f}}}$, "
      rf"$\Omega_\Lambda={ol_med:.3f}^{{+{ol_hi - ol_med:.3f}}}_{{-{ol_med - ol_lo:.3f}}}$"
   )
   if save_path is None:
      save_path = f"corner_plot_om{om_med:.3f}_ol{ol_med:.3f}.png"
   plt.savefig(save_path)

def main():
   parser = ap.ArgumentParser()
   parser.add_argument('--prior_range', type=float, nargs=4, required=True, help='Range for the flat prior in the form Omegam_min Omegam_max Omegal_min Omegal_max')
   parser.add_argument('--nsteps', type=int, required=True, help='Number of MCMC steps')
   parser.add_argument('--nchains', type=int, required=True, help='Number of MCMC chains')
   parser.add_argument('--burnin', type=int, required=True, help='Number of burn-in steps')
   parser.add_argument('--save_path', type=str, default=None, help='Path to save the corner plot')

   args, unknown = parser.parse_known_args()
   prior_range = args.prior_range
   nsteps = args.nsteps
   nchains = args.nchains
   burn_in = args.burnin

   #load Union2.1 z, mu, sigma_mu and covariance
   data = load_union21("data", include_systematics=True)
   print(f"Loaded {len(data)} supernovae from Union2.1")

   #initial posterior to tune scale and covariance of the proposal
   log_prior = flat_log_prior([(prior_range[0], prior_range[1]), (prior_range[2], prior_range[3])])
   likelihood = Likelihood(data, H0=70.0)
   posterior = Posterior(likelihood, log_prior)
   theta_0 = np.array([0.3, 0.7]) #fiducial cosmology for faster convergence
   cov = np.eye(2) #initial guess for proposal covariance
   tuned_result = TuneProp.tune_proposal(posterior, theta_0, cov, n_rounds=5, n_steps_per_round=2000, target_acceptance=(0.2, 0.4), jitter=1e-6)
   tuned_proposal = tuned_result.proposal
   print("\nFrozen proposal covariance:\n", tuned_proposal.cov)

   starts = initialize_chains(nchains, prior_range, posterior)
   print("Initialized walker starting points:\n", starts)

   #run the MCMC sampler with the initialized walkers
   sampler = MetropolisHastings(posterior, tuned_proposal)
   results = [sampler.sample(theta0=s0, n_steps=nsteps, rng=seed) for seed, s0 in enumerate(starts)]
   for i, r in enumerate(results):
       print(f"chain {i}: start={starts[i]}, acceptance={r.acceptance_rate:.2f}")

   om_chains = np.array([r.samples[burn_in:, 0] for r in results])
   ol_chains = np.array([r.samples[burn_in:, 1] for r in results])
   accepted = np.array([r.accepted[burn_in:] for r in results])

   report = convergence_report(
       chains={"omega_m": om_chains, "omega_lambda": ol_chains},
       accepted=accepted,
   )
   print('Rhat:', {k: f"{v:.3f}" for k, v in report.rhat.items()})
   print('ESS:', {k: f"{v:.1f}" for k, v in report.ess.items()})
   print('Acceptance rate:', f"{report.acceptance_rate:.3f}")

   if not report.ok:
       print("Convergence check failed -- refusing to report results from this run.")
       sys.exit(1)

   plot_contours(om_chains, ol_chains, save_path=args.save_path)
   
   print("Convergence check passed")

if __name__ == "__main__":
    main()
