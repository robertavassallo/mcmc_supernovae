# mcmc_supernovae

MCMC pipeline to infer the 2D posterior on $(\Omega_m, \Omega_\Lambda)$ from a Type Ia 
supernova Hubble diagram (Union2.1), without assuming a flat universe — curvature is constrained by
$\Omega_k = 1 - \Omega_m - \Omega_\Lambda$.

## Layout

```
lib/
  cosmology.py     FLRW(H0, omega_m, omega_lambda): E(z), comoving/transverse/
                    luminosity distance, distance modulus mu(z)
  dataset.py       SNDataset, load_union21: loads mu_z.txt and the data covariance matrix
  bayes.py         Posterior, Likelihood (analytic M-marginalisation), flat_log_prior,
                    GaussianRandomWalk proposal, MetropolisHastings sampler,
                    TuneProp.tune_proposal (pilot-tunes and freezes the proposal)
  diagnostics.py   autocorrelation/ESS, split-Rhat, power-spectrum (Dunkley) fit,
                    convergence_report, trace/acceptance plots
scripts/
  run_mcmc.py      end-to-end Union2.1 analysis
data/
  mu_z.txt             name, z, mu, sigma_mu (Union2.1 compilation, 580 SNe)
  cov_matrix_sist.txt  statistical and systematic covariance
notebooks/
  acceptance_tests.ipynb   acceptance tests and analysis notebook
pyproject.toml     packages lib as an installable library 
```

## Model and assumptions

- **Background cosmology:**  
  `FLRW.E(z)` implements $E(z)^2 = \Omega_m(1+z)^3 + \Omega_k(1+z)^2 + \Omega_\Lambda$ with $\Omega_k = 1-\Omega_m-\Omega_\Lambda$
  
- **Absolute-magnitude zero point $M$ is marginalized analytically**.  
  Assuming a flat improper prior on $M$, $\chi^2(\Omega_m,\Omega_\Lambda) = A - B^2/D$, where $A = d^TC^{-1}d$, $B = d^TC^{-1}\mathbf{1}$, $D = \mathbf{1}^TC^{-1}\mathbf{1}$, and $d = \mu_\mathrm{obs} - \mu_\mathrm{th}$.
  
- **$H_0$ is fixed.**  
  `Likelihood` takes `H0=70.0` by default. $D_M \propto c/H_0$. Changing $H_0$ rescales $\mu_\mathrm{th}$ by the same additive constant independent on z:  
  $\mu_\mathrm{th}(z;\Omega_m,\Omega_\Lambda,H_0) = \mu_\mathrm{th}(z;\Omega_m,\Omega_\Lambda,H_0^\mathrm{ref}) + 5\log_{10}(H_0^\mathrm{ref}/H_0)$.  
  Substituting $d \to d - c\mathbf{1}$ for any $c$ into $\chi^2 = A - B^2/D$ (with $A,B,D$ as defined above) leaves it unchanged: for every value of $H_0$, the surface $\chi^2(\Omega_m,\Omega_{\Lambda})$ is the same; Therefore, the `H0=70.0` choice does not affect the fitted $(\Omega_m,\Omega_{\Lambda})$.  
    
- **Inverse covariances are obtained via Cholesky factor.**  
  `cho_factor` computes $C = LL^T$, the Cholesky decomposition of the covariance matrix (which is symmetric positive-definite).  
  $C^{-1}$ applied to a vector $v$ — e.g. $C^{-1}d$ or $C^{-1}\mathbf{1}$ — is then obtained by `cho_solve`, which solves:  
  $Ly = v$ by forward substitution (top row is solved first and since $L$ is lower triangular, row $i$ only involves $y_1,\dots,y_i$, so $y_i = \left(v_i - \sum_{j<i} L_{ij}y_j\right)/L_{ii}$ is computed one at a time in order, each $y_i$ using only the $y_j$'s already found)  
  and $L^Tx = y$ by back substitution (bottom row is solver first and since $L^T$ is upper triangular, row $i$ only involves $x_i,\dots,x_n$, so $x_i = \left(y_i - \sum_{j>i} L^T_{ij}x_j\right)/L^T_{ii}$ is computed from $x_n$ down to $x_1$), giving $x = C^{-1}v$.  
    
- **Flat priors**  
  defined by specified bounds on $\Omega_m$ and $\Omega_{\Lambda}$ independently (`flat_log_prior`)  
  
- **Proposal tuning (warm-up)**:  
  `TuneProp.tune_proposal` runs short MCMC chains from one starting point (chosen near the fiducial values ($\Omega_m$, $\Omega_{\Lambda}$) = (0.3, 0.7)). 
  Each round:  
  1. re-estimates the proposal's covariance from that round's final half (post-transient)
     samples (picking up the $\Omega_m$/$\Omega_{\Lambda}$ correlation),
  2. regularises it with a small `jitter * I` before the Cholesky factorisation,
  3. rescales it by the Roberts-Gelman-Gilks-Rosenthal asymptotically optimal factor $2.38^2/d$.

  The last round's proposal is frozen and used for every chain
  
- **Convergence**  
  values considered for the assumption of convergence for the parameters are split-$\hat R < 1.01$, $\mathrm{ESS} \geq 200$, and acceptance rate within $[0.15, 0.5]$;  
This is checked by `convergence_report`. If any parameter fails the convergence checks, `run_mcmc.py` exits with a non-zero status.  
  

## Data structure

- `mu_z.txt` is the Union2.1 compilation. **`load_union21`** reads `name, z, mu, sigma_mu` (columns 1-4).
- `cov_matrix_sist.txt` (580x580) is the covariance matrix with systematics of data in `mu_z.txt`


## Setup

```bash
cd mcmc_supernovae
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, scipy, matplotlib, pytest
```

### Installation of `lib` as a library

`lib` is a installable package (`pyproject.toml`)  
install it in editable mode to make `import lib.cosmology`,
`lib.bayes`, `lib.dataset`, `lib.diagnostics` available from anywhere:

```bash
pip install -e .
```

## Tests

Tests are performed in `notebooks/acceptance_tests.ipynb` in the following order:

1. **Cosmology core:** $E(0)=1$, low-$z$ Hubble law, $D_M=\chi$ at $\Omega_k=0$, and $D_L=(1+z)^2D_A$ identity.
2. **Bayesian core:** checks if sampler recovers a known Gaussian target's mean and covariance and is reproducible given a seed.
3. **Union2.1 analysis:** checks covariance dimensions/positive-definiteness, $\chi^2$/N for near fiducial parameters, proposal tuning, and dispersed chains.
4. **Convergence report:** trace,acceptance and windowed-mean plots, split-$\hat R$/ESS/acceptance-rate checks, Dunkley power-spectrum diagnostic, and the final $(\Omega_m,\Omega_\Lambda)$ corner plot with median $\pm$ 68% errors.

## Usage: `scripts/run_mcmc.py`

Run from the repository root (`mcmc_supernovae/`):

```bash
python scripts/run_mcmc.py \
  --prior_range Omegam_min Omegam_max Omegal_min Omegal_max \
  --nsteps N --nchains M --burnin B [--save_path PATH]
```

| Argument | Meaning |
|---|---|
| `--prior_range` | Flat-prior bounds, as 4 floats: `Omegam_min Omegam_max Omegal_min Omegal_max` |
| `--nsteps` | Number of MCMC steps per chain |
| `--nchains` | Number of dispersed chains |
| `--burnin` | Steps discarded from the start of every chain before the convergence check |
| `--save_path` | Path to save the corner plot (default: `corner_plot_om<value>_ol<value>.png` in the current directory) |

What it does:

1. Loads the 580-SN Union2.1 dataset (`include_systematics=True`).
2. Builds a flat-prior `Posterior` over `--prior_range` box.
3. Pilot-tunes a `GaussianRandomWalk` proposal (5 rounds x 2000 steps, target
   acceptance 0.2-0.4) starting from a fixed fiducial pilot point
   $(\Omega_m,\Omega_\Lambda)=(0.3,0.7)$ with an initial identity covariance
   guess, and freezes the result.
4. Picks `--nchains` dispersed starting points inside the prior box via Latin
   Hypercube sampling.
5. Runs each chain for `--nsteps` steps with the frozen proposal, then
   discards the first `--burnin` steps of each.
6. Computes `convergence_report`; on failure, prints why and exits with
   status 1 without producing a plot.
7. If successful, saves a corner plot (marginal histograms + 68%/95% joint
   contour, with median $\pm$ 68% percentile errors) to `--save_path`.

Example:

```bash
python scripts/run_mcmc.py --prior_range 0.0 2.0 -1.0 3.0 \
  --nsteps 20000 --nchains 4 --burnin 200 --save_path contours.png
```
