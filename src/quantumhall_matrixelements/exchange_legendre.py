"""Exchange kernels via Gauss-Legendre quadrature with rational mapping."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.special as sps
from scipy.special import roots_legendre

from .exchange_laguerre import _logfact as _laguerre_logfact
from .exchange_laguerre import _precompute_R_table

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]
    BoolArray = NDArray[np.bool_]
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
    G_mags: RealArray      # (nG,)
    G_angles: RealArray    # (nG,)
    z: RealArray           # (nquad,)
    w: RealArray           # (nquad,)
    q_nodes: RealArray     # (nquad,)
    arg: RealArray         # (nG, nquad)
    is_coulomb: bool
    mapped_weight: RealArray  # (nquad,)
    extra_sign_nm: Int8Array # (nmax,nmax)
    d_nm: IntArray        # (nmax,nmax)
    D_nm: IntArray        # (nmax,nmax)
    R_nm: RealArray       # (nquad,nmax,nmax)
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
    lag_scale: float
    lag_J0w: RealArray | None      # (nG_lag, nlag) = J_0(arg) * wg
    lag_L_nm: RealArray | None     # (nmax, nlag) = L_p^0(xg) for p=0..nmax-1
    lag_mask: BoolArray | None     # (nG,) bool — True where Gauss-Laguerre is used


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
    q_nodes = np.sqrt(2.0 * z)

    arg = G_magnitudes[:, None] * q_nodes[None, :]

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

    if is_coulomb:
        mapped_weight = (float(kappa) / q_nodes).astype(np.float64, copy=False)
    elif is_constant:
        mapped_weight = ((float(kappa) / (2.0 * np.pi)) * np.ones_like(z)).astype(
            np.float64, copy=False
        )
    else:
        qvals = q_nodes  # already in 1/ell units supplied by caller
        assert pot_fn is not None
        Vraw = np.asarray(pot_fn(qvals))
        if np.iscomplexobj(Vraw):
            raise ValueError("Callable potential must be real-valued.")
        Vraw = Vraw.astype(np.float64, copy=False)
        if Vraw.shape != z.shape:
            raise ValueError("Callable potential must return array of shape (nquad,)")
        mapped_weight = (Vraw / (2.0 * np.pi)).astype(np.float64, copy=False)

    # Stable radial form-factor table:
    #   R[n,m](q) = sqrt(p!/(p+d)!) * z^(d/2) * L_p^d(z) * exp(-z/2), z=q^2/2
    logfact = _laguerre_logfact(int(nmax)).astype(np.float64)
    R_nm = _precompute_R_table(q_nodes.astype(np.float64, copy=False), logfact)

    # Index-derived arrays
    idx = np.arange(nmax, dtype=int)
    n_idx, m_idx = np.meshgrid(idx, idx, indexing="ij")
    d_nm = np.abs(n_idx - m_idx)
    D_nm = n_idx - m_idx

    extra_sign_nm = (1 - 2 * ((n_idx - m_idx) & 1)).astype(np.int8)  # (-1)^(n-m)

    max_d_sum = 2 * (nmax - 1)
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
    lag_mask: BoolArray | None = None
    lag_scale = float(kappa) / np.sqrt(2.0)
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
        G_mags=G_magnitudes,
        G_angles=G_angles,
        z=z,
        w=w,
        q_nodes=q_nodes,
        arg=arg,
        is_coulomb=is_coulomb,
        mapped_weight=mapped_weight,
        extra_sign_nm=extra_sign_nm,
        d_nm=d_nm,
        D_nm=D_nm,
        R_nm=R_nm,
        parity=parity,
        phase_table=phase_table,
        phase_power=phase_power.astype(np.complex128),
        Ns=Ns,
        minN=minN,
        maxD=maxD,
        J_cache={},
        lag_scale=lag_scale,
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
            radial[pre.lag_mask] = pre.lag_scale * (pre.lag_J0w @ lag_integrand)
            gl_mask = ~pre.lag_mask
            if np.any(gl_mask):
                term = pre.mapped_weight * pre.R_nm[:, n1, m1] * pre.R_nm[:, m2, n2]
                radial[gl_mask] = (J_abs[gl_mask] * pre.w[None, :]) @ term
        else:
            # Standard mapped Gauss-Legendre path
            term = pre.mapped_weight * pre.R_nm[:, n1, m1] * pre.R_nm[:, m2, n2]
            radial = (J_abs * pre.w[None, :]) @ term  # (nG,)

        phase_factor = pre.phase_power[ds]
        pref = phase_factor

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
