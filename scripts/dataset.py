"""Loader for the Union2.1 supernova compilation (mu(z) table + covariance)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

@dataclass
class SNDataset:
    name: np.ndarray
    z: np.ndarray
    mu: np.ndarray
    sigma_mu: np.ndarray
    cov: np.ndarray

    def __len__(self) -> int:
        return len(self.z)

def load_union21(data_dir: Union[str, Path], include_systematics: bool = True) -> SNDataset:
    """Load `mu_z.txt` (name, z, mu, sigma_mu) and the matching covariance matrix.

    Files in data_dir:
      - mu_z.txt: one # header comment line, then one supernova per row
      - cov_matrix.txt: statistical-only covariance
      - cov_matrix_sist.txt: statistical + systematic covariance (default)
    """
    data_dir = Path(data_dir)

    rows = np.genfromtxt(data_dir / "mu_z.txt", comments="#", dtype=None, encoding=None)
    name = np.array([row[0] for row in rows])
    z = np.array([row[1] for row in rows], dtype=float)
    mu = np.array([row[2] for row in rows], dtype=float)
    sigma_mu = np.array([row[3] for row in rows], dtype=float)

    cov_file = "cov_matrix_sist.txt" if include_systematics else "cov_matrix.txt"
    cov = np.loadtxt(data_dir / cov_file, comments="#")

    if cov.shape != (len(z), len(z)):
        raise ValueError(
            f"covariance shape {cov.shape} does not match number of supernovae {len(z)}"
        )

    return SNDataset(name=name, z=z, mu=mu, sigma_mu=sigma_mu, cov=cov)
