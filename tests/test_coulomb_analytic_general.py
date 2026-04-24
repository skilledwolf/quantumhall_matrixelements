from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from quantumhall_matrixelements import get_exchange_kernels_compressed
from quantumhall_matrixelements.exchange_laguerre import (
    QuadratureParams,
    build_exchange_fock_precompute,
)


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


# ---------------------------------------------------------------------------
# Public-API analytic tests.
#
# The tests above validate the internal ``build_exchange_fock_precompute`` +
# ``_numeric_X_element`` pipeline — i.e., the phase tables themselves. They do
# NOT touch the production kernel paths (``get_exchange_kernels_*``).
#
# The tests below close that gap: they compare the **public** compressed
# exchange kernels (via ``get_exchange_kernels_compressed``) against the
# analytic reference for both ``sign_magneticfield = +1`` and ``-1``, for every
# backend. The contract is:
#
#     get_exchange_kernels(..., sign_magneticfield=σ) == X^σ_analytic
#
# with the same σ on both sides. This is the contract the docs advertise and
# the contract every downstream user relies on.
# ---------------------------------------------------------------------------

_BACKENDS = ("laguerre", "ogata", "hankel")


@pytest.mark.parametrize("method", _BACKENDS)
@pytest.mark.parametrize("sign_magneticfield", [-1, +1])
def test_public_exchange_kernels_match_analytic(method, sign_magneticfield):
    """Public API must produce kernels matching analytic σ = sign_magneticfield.

    This test exercises the full production path (``get_exchange_kernels_compressed``)
    for each of the three backends and both sign conventions, and compares
    against the high-precision mpmath analytic reference at the **same** σ.
    """
    nmax = 4                      # keep all 3 backends fast
    kappa = 1.0
    rng = np.random.default_rng(42)

    # A handful of (|G|, θ) points covering G=0, small-G, and large-G regimes.
    G_mags = np.array([0.0, 0.7, 1.8, 3.5, 8.0, 15.0], dtype=np.float64)
    G_thetas = rng.uniform(0.0, 2.0 * np.pi, size=G_mags.size).astype(np.float64)

    # Enumerate all symmetry-canonical (n1, m1, n2, m2); this is small at nmax=4.
    select_list = []
    for n1 in range(nmax):
        for m1 in range(nmax):
            for n2 in range(nmax):
                for m2 in range(nmax):
                    select_list.append((n1, m1, n2, m2))

    values, returned_select = get_exchange_kernels_compressed(
        G_mags,
        G_thetas,
        nmax,
        method=method,
        sign_magneticfield=sign_magneticfield,
        select=select_list,
    )
    assert returned_select == select_list

    ref = _AnalyticCoulombReference(nmax, dps=80)

    # Tolerances chosen to (a) catch any σ-sign bug (which produces errors of
    # order |X|, i.e. O(1)), while (b) tolerating the different backends'
    # intrinsic quadrature noise at small matrix elements. Hankel in particular
    # has ~1e-8 noise at default hankel_N/hankel_h.
    rtol = 1e-6
    atol = 5e-8
    worst = None
    for j, (n1, m1, n2, m2) in enumerate(select_list):
        for g in range(G_mags.size):
            X_num = complex(values[g, j])
            X_ref = ref.X_element(
                n1, m1, n2, m2,
                float(G_mags[g]), float(G_thetas[g]),
                sigma=sign_magneticfield,
                kappa=kappa,
            )
            abs_err = abs(X_num - X_ref)
            tol = atol + rtol * abs(X_ref)
            if abs_err > tol:
                worst = (g, G_mags[g], G_thetas[g],
                         n1, m1, n2, m2, X_num, X_ref, abs_err, tol)
                break
        if worst is not None:
            break

    if worst is not None:
        g, G, theta, n1, m1, n2, m2, X_num, X_ref, abs_err, tol = worst
        raise AssertionError(
            f"Public API disagrees with analytic σ={sign_magneticfield:+d} "
            f"reference for method={method!r}:\n"
            f"  g={g}, |G|={float(G):.6g}, theta={float(theta):.6g}\n"
            f"  (n1,m1,n2,m2)=({n1},{m1},{n2},{m2})\n"
            f"  X_num={X_num}\n"
            f"  X_ref={X_ref}\n"
            f"  |Δ|={abs_err:.3e}, tol={tol:.3e} (rtol={rtol:g}, atol={atol:g})"
        )


@pytest.mark.parametrize("sign_magneticfield", [-1, +1])
def test_public_exchange_kernels_cross_backend_consistent(sign_magneticfield):
    """All three backends must produce identical kernels for the same σ.

    Guards against a single backend silently drifting from the shared
    sign-magneticfield convention (e.g., a change in one backend that flips
    σ for that backend only).
    """
    nmax = 3
    rng = np.random.default_rng(7)
    G_mags = np.array([0.0, 1.2, 4.5, 12.0], dtype=np.float64)
    G_thetas = rng.uniform(0.0, 2.0 * np.pi, size=G_mags.size).astype(np.float64)

    select_list = [
        (n1, m1, n2, m2)
        for n1 in range(nmax)
        for m1 in range(nmax)
        for n2 in range(nmax)
        for m2 in range(nmax)
    ]

    ref_values, _ = get_exchange_kernels_compressed(
        G_mags, G_thetas, nmax,
        method="laguerre",
        sign_magneticfield=sign_magneticfield,
        select=select_list,
    )
    for method in ("ogata", "hankel"):
        values, _ = get_exchange_kernels_compressed(
            G_mags, G_thetas, nmax,
            method=method,
            sign_magneticfield=sign_magneticfield,
            select=select_list,
        )
        diff = float(np.max(np.abs(values - ref_values)))
        # Backends use different quadrature strategies, so use a moderate tol.
        assert diff < 1e-6, (
            f"Backend {method!r} disagrees with laguerre at "
            f"sign_magneticfield={sign_magneticfield}: max|Δ|={diff:.3e}"
        )


@pytest.mark.parametrize("sign_magneticfield", [-1, +1])
def test_form_factors_and_exchange_kernels_share_convention(sign_magneticfield):
    """``get_form_factors`` and ``get_exchange_kernels`` must use the same σ.

    The cross-module consistency contract: form factors at σ=X and the
    exchange-kernel tensor at σ=X must describe the same magnetic-field sign.
    A concrete check: the on-diagonal exchange kernel ``X_{n n n n}(G)`` and
    ``X_{m n m n}(G)`` are built from F^σ_{n,n}(q) and F^σ_{m,n}(q); any σ
    mismatch between the two modules would show up in their agreement with the
    analytic formula (which uses a single σ throughout).
    """
    from quantumhall_matrixelements import get_exchange_kernels, get_form_factors

    nmax = 3
    G_mags = np.array([1.5, 3.0], dtype=np.float64)
    G_thetas = np.array([0.4, -1.2], dtype=np.float64)

    # If F and X use different σ, then F(sm=X) * conj(F(sm=X)) integrated against
    # the Coulomb kernel should NOT reproduce X(sm=X). We check this indirectly
    # by verifying both agree with the analytic formula at the SAME σ.
    X = get_exchange_kernels(
        G_mags, G_thetas, nmax,
        method="laguerre", sign_magneticfield=sign_magneticfield,
    )
    F = get_form_factors(G_mags, G_thetas, nmax, sign_magneticfield=sign_magneticfield)

    # F[g, n', n] phase at off-diagonal entries is i^|n'-n| exp(iσ(n'-n)θ).
    # Verify this matches σ=sign_magneticfield (no inversion).
    for g in range(G_mags.size):
        theta = float(G_thetas[g])
        expected = 1j * np.exp(1j * float(sign_magneticfield) * theta)
        actual_phase = F[g, 1, 0] / abs(F[g, 1, 0])
        np.testing.assert_allclose(
            actual_phase, expected / abs(expected),
            rtol=1e-10, atol=1e-12,
            err_msg=f"form-factor σ != sign_magneticfield at g={g}",
        )

    # Analytic cross-check on a single off-diagonal exchange-kernel entry.
    ref = _AnalyticCoulombReference(nmax, dps=60)
    X_ref_01_10 = complex(ref.X_element(
        0, 1, 1, 0, float(G_mags[0]), float(G_thetas[0]),
        sigma=sign_magneticfield, kappa=1.0,
    ))
    np.testing.assert_allclose(
        complex(X[0, 0, 1, 1, 0]), X_ref_01_10,
        rtol=5e-7, atol=1e-9,
        err_msg=(
            f"Exchange kernel at sign_magneticfield={sign_magneticfield} does NOT "
            f"match analytic σ={sign_magneticfield}; the library is using the "
            f"opposite convention from its form factors."
        ),
    )
