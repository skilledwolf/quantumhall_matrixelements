"""Exchange kernels via Gauss-Legendre quadrature with rational mapping."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.special as sps
from scipy.special import roots_legendre

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]
    IntArray = NDArray[np.int64]
    Int8Array = NDArray[np.int8]

from ._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES, normalize_select


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
@cache
def _logfact(n: int) -> float:
    return float(sps.gammaln(n + 1))


def _parity_factor(N: int) -> int:
    """(-1)^((N-|N|)/2) → (-1)^N for N<0, and 1 for N>=0."""
    return int((-1) ** ((N - abs(N)) // 2))


@cache
def _legendre_nodes_weights_mapped(nquad: int, scale: float) -> tuple[RealArray, RealArray]:
    """
    Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, inf).
    Mapping: z = scale * (1+x)/(1-x)
    Jacobian: dz/dx = scale * 2/(1-x)^2
    """
    x, w_leg = roots_legendre(int(nquad))
    denom = 1.0 - x
    z = scale * (1.0 + x) / denom
    w = w_leg * (scale * 2.0 / (denom * denom))
    return z.astype(np.float64, copy=False), w.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Precompute container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegendrePrecompute:
    nmax: int
    kappa: float
    G_mags: RealArray      # (nG,)
    G_angles: RealArray    # (nG,)
    z: RealArray           # (nquad,)
    w: RealArray           # (nquad,)
    exp_minus_z: RealArray # (nquad,)
    sqrt2z: RealArray      # (nquad,)
    arg: RealArray         # (nG, nquad)
    is_coulomb: bool
    Veff: RealArray | None # (nquad,) or None
    z_pows: RealArray      # (max_d_sum+1, nquad)
    C_nm: RealArray        # (nmax,nmax)
    extra_sign_nm: Int8Array # (nmax,nmax)
    d_nm: IntArray        # (nmax,nmax)
    D_nm: IntArray        # (nmax,nmax)
    L_nm: RealArray        # (nmax,nmax,nquad)
    parity: Int8Array      # (2*maxD+1,)
    phase_table: ComplexArray # (2*maxD+1, nG)
    phase_power: ComplexArray # (max_d_sum+1,)
    Ns: IntArray          # (-maxD..maxD)
    minN: int
    maxD: int

    # mutable cache for Bessel tables (abs order -> (nG,nquad))
    J_cache: dict[int, RealArray]

    # Gauss-Laguerre quadrature for ds=0 Coulomb (alpha=-0.5).
    # Only used for G values where the Bessel oscillations are resolvable.
    lag_J0w: RealArray | None      # (nG_lag, nlag) = J_0(arg) * wg
    lag_L_nm: RealArray | None     # (nmax, nlag) = L_p^0(xg) for p=0..nmax-1
    lag_mask: RealArray | None     # (nG,) bool — True where Gauss-Laguerre is used


# ---------------------------------------------------------------------------
# Build precompute
# ---------------------------------------------------------------------------
def _build_legendre_precompute(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
    nquad: int = 8000,
    scale: float = 0.5,
    nlag: int = 80,
) -> LegendrePrecompute:
    G_magnitudes = np.asarray(G_magnitudes, dtype=float).ravel()
    G_angles = np.asarray(G_angles, dtype=float).ravel()

    # Quadrature grid and derived arrays
    z, w = _legendre_nodes_weights_mapped(int(nquad), float(scale))
    exp_minus_z = np.exp(-z)
    sqrt2z = np.sqrt(2.0 * z)

    arg = G_magnitudes[:, None] * sqrt2z[None, :]

    # Potential
    if callable(potential):
        pot_kind = "callable"
        pot_fn = potential
    else:
        pot_kind = str(potential).strip().lower()
        pot_fn = None

    if pot_kind not in {"coulomb", "constant", "callable"}:
        raise ValueError("potential must be 'coulomb', 'constant', or a callable V(q)")
    is_coulomb = pot_kind == "coulomb"
    is_constant = pot_kind == "constant"

    Veff = None
    if not is_coulomb:
        if is_constant:
            Veff = (float(kappa) / (2.0 * np.pi)) * np.ones_like(z)
        else:
            qvals = sqrt2z  # already in 1/ell units supplied by caller
            assert pot_fn is not None
            Vraw = np.asarray(pot_fn(qvals))
            if np.iscomplexobj(Vraw):
                raise ValueError("Callable potential must be real-valued.")
            Vraw = Vraw.astype(np.float64, copy=False)
            if Vraw.shape != z.shape:
                raise ValueError("Callable potential must return array of shape (nquad,)")
            Veff = Vraw / (2.0 * np.pi)

    # Index-derived arrays
    idx = np.arange(nmax, dtype=int)
    n_idx, m_idx = np.meshgrid(idx, idx, indexing="ij")
    p_nm = np.minimum(n_idx, m_idx)
    d_nm = np.abs(n_idx - m_idx)
    D_nm = n_idx - m_idx

    extra_sign_nm = (1 - 2 * ((n_idx - m_idx) & 1)).astype(np.int8)  # (-1)^(n-m)

    C_nm = np.empty((nmax, nmax), dtype=np.float64)
    for n in range(nmax):
        for m in range(nmax):
            p = int(p_nm[n, m])
            d = int(d_nm[n, m])
            C_nm[n, m] = np.exp(0.5 * (_logfact(p) - _logfact(p + d)))

    # Laguerre table L_p^d(z)
    max_d_sum = 2 * (nmax - 1)
    L_nm = np.empty((nmax, nmax, z.size), dtype=np.float64)
    lag_cache: dict[tuple[int, int], RealArray] = {}
    for n in range(nmax):
        for m in range(nmax):
            p = int(p_nm[n, m])
            d = int(d_nm[n, m])
            key = (p, d)
            if key not in lag_cache:
                lag_cache[key] = sps.eval_genlaguerre(p, d, z).astype(np.float64)
            L_nm[n, m, :] = lag_cache[key]

    # z powers
    z_pows = np.empty((max_d_sum + 1, z.size), dtype=np.float64)
    if is_coulomb:
        for ds in range(max_d_sum + 1):
            z_pows[ds] = z ** (0.5 * (ds - 1))
    else:
        for ds in range(max_d_sum + 1):
            z_pows[ds] = z ** (0.5 * ds)

    maxD = 2 * (nmax - 1)
    Ns = np.arange(-maxD, maxD + 1, dtype=int)
    minN = int(Ns[0])
    parity = np.array([_parity_factor(int(N)) for N in Ns], dtype=np.int8)
    phase_table = np.exp(-1j * Ns[:, None] * G_angles[None, :])
    phase_power = (1j) ** np.arange(max_d_sum + 1, dtype=int)

    # Gauss-Laguerre for ds=0 Coulomb: weight z^{-0.5} e^{-z} absorbs singularity.
    # Only used for |G| small enough that Bessel oscillations are resolvable by
    # the Gauss-Laguerre nodes. For large |G|, fall back to mapped GL.
    lag_J0w: RealArray | None = None
    lag_L_nm: RealArray | None = None
    lag_mask: RealArray | None = None
    if is_coulomb:
        nlag_eff = min(max(int(nlag), 2 * nmax), 350)
        xg, wg = sps.roots_genlaguerre(nlag_eff, -0.5)
        # Threshold: use Gauss-Laguerre only when Bessel oscillations are
        # adequately sampled. At the largest node x_max, the argument of J_0 is
        # G*sqrt(2*x_max), giving ~G*sqrt(2*x_max)/(2π) oscillation cycles.
        # We need nlag_eff to comfortably exceed ~8 * n_cycles.
        x_max = float(xg[-1]) if xg.size > 0 else 1.0
        g_threshold = 0.5 * np.pi * np.sqrt(nlag_eff / (2.0 * x_max)) * nlag_eff
        # Simpler conservative threshold: G < sqrt(nlag_eff)
        g_threshold = float(np.sqrt(nlag_eff))
        lag_mask = G_magnitudes <= g_threshold  # (nG,) bool
        G_lag = G_magnitudes[lag_mask]
        if G_lag.size > 0:
            lag_arg = (G_lag[:, None] * np.sqrt(2.0 * xg)[None, :]).astype(np.float64)
            lag_J0w = (sps.jv(0, lag_arg) * wg[None, :]).astype(np.float64)
        else:
            lag_J0w = np.empty((0, nlag_eff), dtype=np.float64)
        # For ds=0, both d1=d2=0, so we only need L_p^0(xg) for p=0..nmax-1
        lag_L_nm = np.empty((nmax, xg.size), dtype=np.float64)
        for p in range(nmax):
            lag_L_nm[p, :] = sps.eval_genlaguerre(p, 0, xg)

    return LegendrePrecompute(
        nmax=nmax,
        kappa=float(kappa),
        G_mags=G_magnitudes,
        G_angles=G_angles,
        z=z,
        w=w,
        exp_minus_z=exp_minus_z,
        sqrt2z=sqrt2z,
        arg=arg,
        is_coulomb=is_coulomb,
        Veff=Veff,
        z_pows=z_pows,
        C_nm=C_nm,
        extra_sign_nm=extra_sign_nm,
        d_nm=d_nm,
        D_nm=D_nm,
        L_nm=L_nm,
        parity=parity,
        phase_table=phase_table,
        phase_power=phase_power.astype(np.complex128),
        Ns=Ns,
        minN=minN,
        maxD=maxD,
        J_cache={},
        lag_J0w=lag_J0w,
        lag_L_nm=lag_L_nm,
        lag_mask=lag_mask,
    )


# ---------------------------------------------------------------------------
# Single-element evaluator using precompute
# ---------------------------------------------------------------------------
def _evaluate_legendre(
    pre: LegendrePrecompute, select_list: list[tuple[int, int, int, int]]
) -> ComplexArray:
    nG = pre.G_mags.size
    values = np.zeros((nG, len(select_list)), dtype=np.complex128)
    sqrt2 = np.sqrt(2.0)

    for idx_sel, (n1, m1, n2, m2) in enumerate(select_list):
        d1 = int(pre.d_nm[n1, m1])
        d2 = int(pre.d_nm[m2, n2])
        ds = d1 + d2
        D1 = int(pre.D_nm[n1, m1])
        D2 = int(pre.D_nm[m2, n2])
        N = D1 + D2
        absN = abs(int(N))

        # Bessel cache per |N|
        if absN not in pre.J_cache:
            pre.J_cache[absN] = sps.jv(absN, pre.arg)
        J_abs = pre.J_cache[absN]

        # Radial integral
        if ds == 0 and pre.is_coulomb and pre.lag_mask is not None and np.any(pre.lag_mask):
            # Hybrid: Gauss-Laguerre for small |G| (singularity-dominated),
            # mapped GL for large |G| (oscillation-dominated).
            assert pre.lag_J0w is not None and pre.lag_L_nm is not None
            radial = np.empty(nG, dtype=np.float64)
            L1_lag = pre.lag_L_nm[n1]  # (nlag,)  — p=n1, d=0
            L2_lag = pre.lag_L_nm[m2]  # (nlag,)  — p=m2, d=0
            lag_integrand = L1_lag * L2_lag  # (nlag,)
            radial[pre.lag_mask] = pre.lag_J0w @ lag_integrand
            gl_mask = ~pre.lag_mask
            if np.any(gl_mask):
                term = pre.exp_minus_z * pre.z_pows[ds] * pre.L_nm[n1, m1] * pre.L_nm[m2, n2]
                radial[gl_mask] = (J_abs[gl_mask] * pre.w[None, :]) @ term
        else:
            # Standard mapped Gauss-Legendre path
            term = pre.exp_minus_z * pre.z_pows[ds] * pre.L_nm[n1, m1] * pre.L_nm[m2, n2]
            if not pre.is_coulomb:
                term = term * pre.Veff
            radial = (J_abs * pre.w[None, :]) @ term  # (nG,)

        C1 = pre.C_nm[n1, m1]
        C2 = pre.C_nm[n2, m2]
        phase_factor = pre.phase_power[ds]
        pref = (C1 * C2) * phase_factor
        if pre.is_coulomb:
            pref = (float(pre.kappa) * pref) / sqrt2

        extra = pre.extra_sign_nm[m2, n2]
        N_idx = int(N - pre.minN)
        signN = pre.parity[N_idx]
        phase_N = pre.phase_table[N_idx]

        values[:, idx_sel] = phase_N * radial * (pref * extra * signN)

    return values


# ---------------------------------------------------------------------------
# Public backend API
# ---------------------------------------------------------------------------
def get_exchange_kernels_GaussLegendre(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
    nquad: int = 8000,
    scale: float = 0.5,
    nlag: int = 80,
    sign_magneticfield: int = -1,
    select: Iterable[tuple[int, int, int, int]] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
) -> tuple[ComplexArray, list[tuple[int, int, int, int]]]:
    """Compute exchange kernels using mapped Gauss–Legendre quadrature.

    Returns compressed values and the associated select_list. Use the public
    dispatcher in ``__init__.py`` to optionally materialize the full tensor.
    """

    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")

    G_magnitudes = np.asarray(G_magnitudes, dtype=float).ravel()
    G_angles = np.asarray(G_angles, dtype=float).ravel()
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")

    nmax = int(nmax)
    if nmax <= 0:
        raise ValueError("nmax must be positive")

    select_list, sel_n1, sel_m1, sel_n2, sel_m2 = normalize_select(
        nmax, select, canonical_select_max_entries=canonical_select_max_entries
    )

    # Build and evaluate
    pre = _build_legendre_precompute(
        G_magnitudes,
        G_angles,
        nmax,
        potential=potential,
        kappa=kappa,
        nlag=nlag,
        nquad=nquad,
        scale=scale,
    )

    vals = _evaluate_legendre(pre, select_list)

    if sign_magneticfield == 1:
        phase1 = 1 - 2 * ((sel_n1 - sel_m1) & 1)
        phase2 = 1 - 2 * ((sel_n2 - sel_m2) & 1)
        vals = np.conj(vals) * (phase1 * phase2)[None, :]

    return vals, select_list


__all__ = ["get_exchange_kernels_GaussLegendre"]
