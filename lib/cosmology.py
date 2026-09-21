"""FLRW background cosmology: expansion rate, distances and distance modulus.

    E(z)^2 = Omega_m (1+z)^3 + Omega_k (1+z)^2 + Omega_lambda
    Omega_k = 1 - Omega_m - Omega_lambda            (no flatness assumed)

    D_H = c / H0                                     (Hubble distance)
    chi(z) = D_H * integral_0^z dz' / E(z')           (comoving distance)
    D_M(z) = chi                                                     if Omega_k == 0
             D_H/sqrt(Omega_k) * sinh(sqrt(Omega_k) * chi/D_H)        if Omega_k > 0
             D_H/sqrt(|Omega_k|) * sin(sqrt(|Omega_k|) * chi/D_H)     if Omega_k < 0
    D_L(z) = (1 + z) * D_M(z)                          (luminosity distance)
    mu(z) = 5 log10(D_L [Mpc]) + 25                    (distance modulus)
    t(z) = 1/H0 * integral_z^inf dz' / ((1+z') E(z'))     (age of the universe at z)
    eta(z) = 1/H0 * integral_z^inf dz' / E(z')             (conformal time since the Big Bang)
"""
from __future__ import annotations

import numpy as np
import scipy.integrate

C_KM_S = 299792.458  #c in km/s
GYR_KMS_MPC = 977.792  # 1/H0 in Gyr for H0 in km/s/Mpc (1 Mpc/(km/s) in Gyr)

class FLRW:
    def __init__(self, H0: float, omega_m: float, omega_lambda: float):
        self.H0 = float(H0)
        self.omega_m = float(omega_m)
        self.omega_lambda = float(omega_lambda)
        self.omega_k = 1.0 - self.omega_m - self.omega_lambda

    def E(self, z):
        """H(z)/H0
        Input: scalar or np array of z
        """
        return np.sqrt(self.omega_m*(1 + z)**3 + self.omega_k*(1 + z)**2 + self.omega_lambda)

    def H(self, z):
        """Hubble parameter H(z) in km/s/Mpc."""
        return self.H0*self.E(z)

    def chi(self, z):
        """comoving distance chi(z) in Mpc
        Input: scalar or np array of z
        Calculates chi(z) via an interpolator over the integral of 
        1/E(z') computed on a fine z' grid in order to not compute 
        one integral for each z value
        """
        z_grid = np.linspace(0, np.max(np.atleast_1d(z)), 2000)
        integrand = 1/self.E(z_grid)
        cum = np.concatenate(([0.0], scipy.integrate.cumulative_trapezoid(integrand, z_grid)))
        chi = C_KM_S/self.H0*np.interp(np.atleast_1d(z), z_grid, cum)
        return chi if np.ndim(z) > 0 else chi[0]

    def D_M(self, z):
        """Transverse comoving distance D_M(z) corrected for curvature in Mpc."""
        if self.omega_k == 0:
            return self.chi(z)
        elif self.omega_k > 0:
            return (C_KM_S/self.H0)/np.sqrt(self.omega_k) * np.sinh(np.sqrt(self.omega_k) * self.chi(z)/(C_KM_S/self.H0))
        else:
            return (C_KM_S/self.H0)/np.sqrt(np.abs(self.omega_k)) * np.sin(np.sqrt(np.abs(self.omega_k)) * self.chi(z)/(C_KM_S/self.H0))

    def D_A(self, z):
        """Angular diameter distance D_A(z) = D_M(z)/(1+z) in Mpc."""
        return self.D_M(z)/(1 + z)

    def D_L(self, z):
        """Luminosity distance D_L(z) = (1+z) D_M(z), in Mpc."""
        return (1 + z) * self.D_M(z)

    def mu(self, z):
        """Distance modulus mu(z) = 5 log10(D_L[Mpc]) + 25 in magnitudes."""
        return 5*np.log10(self.D_L(z)) + 25

    def dV_dz_dOmega(self, z):
        """Comoving volume element: comoving volume lying in 
        a redshift shell dz and solid angle dΩ
        dV/dzdOmega = D_H D_M(z)^2 / E(z), in Mpc^3/sr.
        """
        return (C_KM_S/self.H0) * self.D_M(z)**2 / self.E(z)
    
    def t(self, z):
        """Age of the universe at redshift z, in Gyr.
        Input: scalar or np array of z
        t(z) = t(0) - lookback_time(z), where lookback_time(z) = 1/H0 * integral_0^z dz'/((1+z') E(z'))
        is computed via an interpolator over the integral of 1/((1+z') E(z'))
        """
        #age of universe today
        integrand_1 = lambda zp: 1.0/((1 + zp)*self.E(zp))
        val, _ = scipy.integrate.quad(integrand_1, 0.0, np.inf)
        t0 = val*GYR_KMS_MPC/self.H0

        z_grid = np.linspace(0, np.max(np.atleast_1d(z)), 2000)
        integrand = 1.0/((1 + z_grid)*self.E(z_grid))
        cum = np.concatenate(([0.0], scipy.integrate.cumulative_trapezoid(integrand, z_grid)))
        lookback_time = GYR_KMS_MPC/self.H0*np.interp(np.atleast_1d(z), z_grid, cum)
        t_of_z = t0 - lookback_time
        return t_of_z if np.ndim(z) > 0 else t_of_z[0]

    def eta(self, z):
        """Conformal time at redshift z, in Gyr.
        eta(z) = eta(0) - integral_0^z dz'/E(z')/H0.
        """
        val, _ = scipy.integrate.quad(lambda zp: 1.0/self.E(zp), 0.0, np.inf)
        eta0 = GYR_KMS_MPC/self.H0*val  # Conformal time at z=0 in Gyr

        eta_lookback = self.chi(z)/C_KM_S
        eta_of_z = eta0 - eta_lookback
        return eta_of_z if np.ndim(z) > 0 else eta_of_z[0]
