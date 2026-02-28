from __future__ import annotations

import mpmath as mp
import numpy as np

from quantumhall_matrixelements.fock_fast import QuadratureParams, build_exchange_fock_precompute


def _numeric_X_element(pre, g: int, n1: int, m1: int, n2: int, m2: int) -> complex:
    R = pre.R
    w = pre.w_eff
    kern = pre.kernels
    ph_in = pre.phase_in
    ph_out = pre.phase_out
    shift = pre.max_d
    max_order = pre.max_order

    d_out = m1 - n1
    d_in = m2 - n2
    out_idx = d_out + shift
    in_idx = d_in + shift
    k = d_in - d_out
    k_idx = k + max_order

    phase = ph_in[g, in_idx] * ph_out[g, out_idx]
    val = phase * np.dot(w * R[:, m1, n1] * R[:, n2, m2], kern[g, :, k_idx])
    return complex(val)


class _AnalyticCoulombReference:
    """High-precision analytic Coulomb reference for X_{n1 m1 n2 m2}(G)."""

    def __init__(self, nmax: int, *, dps: int = 120):
        self.nmax = int(nmax)
        mp.mp.dps = int(dps)
        self.fact = [mp.factorial(i) for i in range(self.nmax)]
        self._lag_cache: dict[tuple[int, int], list[mp.mpf]] = {}
        self._moment_cache: dict[tuple[int, int, float], mp.mpf] = {}

    def laguerre_coeffs(self, n: int, alpha: int) -> list[mp.mpf]:
        key = (int(n), int(alpha))
        if key in self._lag_cache:
            return self._lag_cache[key]
        n_i, a_i = key
        coeffs = [
            (-1) ** j * mp.binomial(n_i + a_i, n_i - j) / self.fact[j] for j in range(n_i + 1)
        ]
        self._lag_cache[key] = coeffs
        return coeffs

    def gauss_bessel_moment(self, k: int, m: int, G: float) -> mp.mpf:
        """I_{k,m}(G) = ∫_0^∞ dq q^m e^{-q^2/2} J_k(Gq) dq."""
        key = (int(k), int(m), float(G))
        if key in self._moment_cache:
            return self._moment_cache[key]

        nu = abs(int(k))
        G_mp = mp.mpf(G)

        if G_mp == 0:
            val = (
                mp.mpf("0")
                if nu != 0
                else 2 ** ((m - 1) / 2) * mp.gamma((m + 1) / 2)
            )
            self._moment_cache[key] = val
            return val

        z = mp.mpf("0.5") * G_mp * G_mp
        a = mp.mpf("0.5") * (nu + m + 1)
        b = nu + 1
        try:
            hyp = mp.hyp1f1(a, b, -z)
        except mp.NoConvergence:
            hyp = mp.hyp1f1(a, b, -z, maxterms=10**6)

        val = (G_mp**nu) * mp.gamma(a) / (2 ** ((nu + 1 - m) / 2) * mp.gamma(nu + 1)) * hyp
        if k < 0 and (nu % 2 == 1):
            val = -val

        self._moment_cache[key] = val
        return val

    def X_element(
        self,
        n1: int,
        m1: int,
        n2: int,
        m2: int,
        G: float,
        theta_G: float,
        *,
        sigma: int,
        kappa: float,
    ) -> complex:
        d_out = int(m1 - n1)
        d_in = int(m2 - n2)
        a_out = abs(d_out)
        a_in = abs(d_in)
        a_tot = a_out + a_in
        k = int(d_in - d_out)

        n1_min, n1_max = (n1, m1) if n1 <= m1 else (m1, n1)
        n2_min, n2_max = (n2, m2) if n2 <= m2 else (m2, n2)

        ratio = mp.sqrt(self.fact[n1_min] / self.fact[n1_max]) * mp.sqrt(
            self.fact[n2_min] / self.fact[n2_max]
        )
        phase = (1j) ** a_tot * ((-1) ** d_in) * mp.e ** (
            1j * int(sigma) * (d_out - d_in) * mp.mpf(theta_G)
        )
        pref = mp.mpf(kappa) * ratio * (2 ** (-mp.mpf(a_tot) / 2)) * phase

        c1 = self.laguerre_coeffs(n1_min, a_out)
        c2 = self.laguerre_coeffs(n2_min, a_in)

        poly = [mp.mpf("0")] * (len(c1) + len(c2) - 1)
        for i, ci in enumerate(c1):
            for j, cj in enumerate(c2):
                poly[i + j] += ci * cj

        S = mp.mpf("0")
        for t, ct in enumerate(poly):
            if ct == 0:
                continue
            m_pow = a_tot + 2 * t
            S += ct * (2 ** (-t)) * self.gauss_bessel_moment(k, m_pow, float(G))

        return complex(pref * S)


def test_coulomb_general_elements_large_nmax_against_analytic():
    # This mirrors playground/benchmark_coulomb_vs_analytic.py but keeps runtime
    # reasonable for pytest by sampling fewer elements.
    nmax = 100
    sigma = 1
    kappa = 1.0

    nG = 11
    Gmax = 30.0
    Nq = 420
    qmax = 30.0

    nsamples = 40
    seed = 1234
    dps = 120

    rng = np.random.default_rng(seed)
    G_mags = np.linspace(0.0, Gmax, nG, dtype=np.float64)
    G_thetas = rng.uniform(0.0, 2.0 * np.pi, size=nG).astype(np.float64)

    pre = build_exchange_fock_precompute(
        nmax=nmax,
        G_mags=G_mags,
        G_thetas=G_thetas,
        params=QuadratureParams(qmax=qmax, N=Nq),
        sigma=sigma,
        kappa=kappa,
        potential=None,  # Coulomb
        include_minus=False,
    )

    ref = _AnalyticCoulombReference(nmax, dps=dps)

    rng2 = np.random.default_rng(seed + 17)
    rtol = 2e-8
    atol = 2e-10

    worst = None
    for _ in range(nsamples):
        g = int(rng2.integers(0, nG))
        n1, m1, n2, m2 = [int(rng2.integers(0, nmax)) for _ in range(4)]
        G = float(G_mags[g])
        theta = float(G_thetas[g])

        X_num = _numeric_X_element(pre, g, n1, m1, n2, m2)
        X_ref = ref.X_element(n1, m1, n2, m2, G, theta, sigma=sigma, kappa=kappa)

        abs_err = abs(X_num - X_ref)
        abs_ref = abs(X_ref)
        tol = atol + rtol * abs_ref
        if abs_err > tol:
            worst = (g, G, theta, n1, m1, n2, m2, X_num, X_ref, abs_err, tol)
            break

    if worst is not None:
        g, G, theta, n1, m1, n2, m2, X_num, X_ref, abs_err, tol = worst
        raise AssertionError(
            "Coulomb analytic validation failed for a random element:\n"
            f"  g={g}, |G|={G:.6g}, theta={theta:.6g}\n"
            f"  (n1,m1,n2,m2)=({n1},{m1},{n2},{m2})\n"
            f"  X_num={X_num}\n"
            f"  X_ref={X_ref}\n"
            f"  |Δ|={abs_err:.3e}, tol={tol:.3e} (rtol={rtol:g}, atol={atol:g})"
        )
