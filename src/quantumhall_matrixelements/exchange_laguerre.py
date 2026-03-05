"""Optimized exchange-Fock application using finite-q Gauss–Legendre quadrature.

This is a high-throughput path for repeated Fock contractions Σ(G) = -X(G)·ρ(G)
that avoids materializing the full 5D exchange tensor.

It mirrors the algorithm in ``playground/qhall_fock_reference_v2.py``:

- finite-q Gauss–Legendre quadrature on [0, qmax]
- precompute radial form-factor pieces R_{n',n}(q)
- precompute Bessel tables J_k(q*|G|) for k in [-2(nmax-1), 2(nmax-1)]
- Toeplitz contraction in the LL-difference index d = m-n

This module requires ``numba``; the core contraction is JIT-compiled for
performance and no pure-NumPy fallback is provided.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache as _cache
from typing import TypeVar, cast

import numpy as np
from numba import njit, prange
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy import special

from ._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES, normalize_select

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_NumbaFunc = TypeVar("_NumbaFunc", bound=Callable[..., object])


def _typed_njit(
    *, parallel: bool = False, fastmath: bool = False
) -> Callable[[_NumbaFunc], _NumbaFunc]:
    """Type-preserving wrapper around ``numba.njit`` for strict mypy."""
    return cast(
        "Callable[[_NumbaFunc], _NumbaFunc]",
        njit(parallel=parallel, fastmath=fastmath),
    )


def cartesian_to_polar(Gxy: RealArray) -> tuple[RealArray, RealArray]:
    """Convert (nG,2) Cartesian G vectors to (|G|, theta)."""
    Gxy = np.asarray(Gxy, dtype=np.float64)
    if Gxy.ndim != 2 or Gxy.shape[1] != 2:
        raise ValueError("Gxy must have shape (nG, 2)")
    Gx = Gxy[:, 0]
    Gy = Gxy[:, 1]
    mags = np.sqrt(Gx * Gx + Gy * Gy)
    thetas = np.arctan2(Gy, Gx)
    return mags, thetas


def _logfact(nmax: int) -> RealArray:
    n = np.arange(nmax + 1, dtype=np.float64)
    return cast("RealArray", special.gammaln(n + 1.0))


def _legendre_q_nodes_weights(N: int, qmax: float) -> tuple[RealArray, RealArray]:
    t, w = leggauss(int(N))
    q = 0.5 * float(qmax) * (t + 1.0)
    wq = 0.5 * float(qmax) * w
    return q.astype(np.float64), wq.astype(np.float64)


# ---------------------------------------------------------------------------
# Ogata q-space quadrature helpers
# ---------------------------------------------------------------------------
def _ogata_psi(t: RealArray) -> RealArray:
    """Ogata's double-exponential map psi(t) = t * tanh((pi/2) sinh t)."""
    return np.asarray(t * np.tanh(0.5 * np.pi * np.sinh(t)), dtype=np.float64)


def _ogata_dpsi(t: RealArray) -> RealArray:
    """Derivative of psi(t), stabilized for large t."""
    t = np.asarray(t, dtype=np.float64)
    out = np.ones_like(t)
    m = t < 6.0
    if np.any(m):
        tt = t[m]
        out[m] = (np.pi * tt * np.cosh(tt) + np.sinh(np.pi * np.sinh(tt))) / (
            1.0 + np.cosh(np.pi * np.sinh(tt))
        )
    return out


@_cache
def _ogata_q_nodes(nu: int, h: float, N: int) -> tuple[RealArray, RealArray]:
    r"""Ogata nodes and base weights for the integral without extra r factor.

    For :math:`\int_0^\infty f(r)\,J_\nu(kr)\,\mathrm{d}r` we approximate

    .. math::
        \approx \sum_n \frac{\mathrm{sf}[n]}{k}\,f\!\bigl(x[n]/k\bigr)

    Returns ``(x_nodes, series_fac)`` where ``series_fac`` does **not** include
    the extra ``x`` factor used in the standard Hankel-transform convention.
    """
    nu = int(nu)
    h = float(h)
    N = int(N)
    roots = special.jn_zeros(nu, N) / np.pi
    t = h * roots
    x = np.pi * _ogata_psi(t) / h
    j = np.pi * roots
    w = special.yv(nu, j) / special.jv(nu + 1, j)
    kernel = special.jv(nu, x)
    series_fac = np.pi * w * kernel * _ogata_dpsi(t)
    return x.astype(np.float64), series_fac.astype(np.float64)


@_typed_njit(fastmath=False)
def _precompute_R_table(q_nodes: RealArray, logfact: RealArray) -> RealArray:
    """Compute R[iq, n, m] such that F_{n,m}(q,phi)=phase*R[iq,n,m]."""
    Nq = q_nodes.size
    nmax = logfact.size - 1
    R = np.empty((Nq, nmax, nmax), dtype=np.float64)

    # Ltab[a, k] = L_k^a(x) for a=0..nmax-1, k=0..nmax-1
    Ltab = np.empty((nmax, nmax), dtype=np.float64)
    pow_x = np.empty(nmax, dtype=np.float64)

    for iq in range(Nq):
        q = q_nodes[iq]
        x = 0.5 * q * q

        for a in range(nmax):
            Ltab[a, 0] = 1.0
            if nmax > 1:
                Ltab[a, 1] = 1.0 + a - x
            for k in range(1, nmax - 1):
                # (k+1) L_{k+1} = (2k+1+a-x)L_k - (k+a)L_{k-1}
                Ltab[a, k + 1] = (
                    (2 * k + 1 + a - x) * Ltab[a, k] - (k + a) * Ltab[a, k - 1]
                ) / (k + 1)

        logx = np.log(x)
        for a in range(nmax):
            pow_x[a] = np.exp(0.5 * a * logx)
        ex = np.exp(-0.5 * x)

        for n in range(nmax):
            for np_ in range(nmax):
                a = np_ - n
                if a < 0:
                    a = -a
                kmin = n if n < np_ else np_
                ratio = np.exp(0.5 * (logfact[kmin] - logfact[kmin + a]))
                R[iq, np_, n] = ratio * pow_x[a] * Ltab[a, kmin] * ex

    return R


def _precompute_bessel_table(G_mags: RealArray, q_nodes: RealArray, max_order: int) -> RealArray:
    """kernels[g, iq, k_shift] = J_k(q_i*|G_g|) for k in [-max_order, +max_order]."""
    G_mags = np.asarray(G_mags, dtype=np.float64)
    q_nodes = np.asarray(q_nodes, dtype=np.float64)
    nG = int(G_mags.size)
    Nq = int(q_nodes.size)

    K = 2 * int(max_order) + 1
    orders = np.arange(int(max_order) + 1, dtype=np.int32)
    z = G_mags[:, None] * q_nodes[None, :]

    pos = special.jv(orders[:, None, None], z[None, :, :]).astype(np.float64, copy=False)
    pos = np.transpose(pos, (1, 2, 0))

    kernels = np.empty((nG, Nq, K), dtype=np.float64)
    kernels[:, :, max_order : max_order + max_order + 1] = pos

    sign = (1.0 - 2.0 * (orders % 2)).astype(np.float64)
    p = np.arange(1, int(max_order) + 1, dtype=np.int32)
    kernels[:, :, max_order - p] = pos[:, :, p] * sign[p]
    return kernels


def _build_phase_tables(
    G_thetas: RealArray, max_d: int, sigma: float
) -> tuple[ComplexArray, ComplexArray]:
    d_vals = np.arange(-max_d, max_d + 1, dtype=np.int32)
    phase_base = (1j) ** np.abs(d_vals)
    phase_base = phase_base.astype(np.complex128)

    phase_base_in = phase_base * ((-1.0) ** d_vals).astype(np.float64)
    phase_base_out = phase_base

    nG = int(G_thetas.size)
    phase_in = np.empty((nG, d_vals.size), dtype=np.complex128)
    phase_out = np.empty_like(phase_in)

    for g in range(nG):
        theta = float(G_thetas[g])
        ang = float(sigma) * theta * d_vals.astype(np.float64)
        epos = np.cos(ang) + 1j * np.sin(ang)
        phase_out[g, :] = phase_base_out * epos
        phase_in[g, :] = phase_base_in * np.conjugate(epos)

    return phase_in, phase_out


@_typed_njit(parallel=True, fastmath=False)
def _exchange_fock_numba(
    rho: ComplexArray,
    R: RealArray,
    w_eff: RealArray,
    kernels: RealArray,
    phase_in: ComplexArray,
    phase_out: ComplexArray,
    max_d: int,
    max_order: int,
) -> ComplexArray:
    nG, nmax, _ = rho.shape
    Nq = R.shape[0]
    D = 2 * max_d + 1
    shift = max_d

    F = np.zeros((nG, nmax, nmax), dtype=np.complex128)

    for g in prange(nG):
        svec = np.empty(D, dtype=np.complex128)
        uvec = np.empty(D, dtype=np.complex128)
        vvec = np.empty(D, dtype=np.complex128)

        for iq in range(Nq):
            for idx in range(D):
                svec[idx] = 0.0 + 0.0j

            Ri = R[iq]
            for n2 in range(nmax):
                for m2 in range(nmax):
                    svec[m2 - n2 + shift] += rho[g, n2, m2] * Ri[n2, m2]

            for idx in range(D):
                uvec[idx] = svec[idx] * phase_in[g, idx]

            ker = kernels[g, iq]
            for out_idx in range(D):
                acc = 0.0 + 0.0j
                for in_idx in range(D):
                    acc += uvec[in_idx] * ker[in_idx - out_idx + max_order]
                vvec[out_idx] = acc * phase_out[g, out_idx]

            w = w_eff[iq]
            for n1 in range(nmax):
                for m1 in range(nmax):
                    F[g, n1, m1] += (w * Ri[m1, n1]) * vvec[m1 - n1 + shift]

    return F


@_typed_njit(parallel=True, fastmath=False)
def _evaluate_exchange_kernels_laguerre_numba(
    R: RealArray,
    w_eff: RealArray,
    kernels: RealArray,
    phase_in: ComplexArray,
    phase_out: ComplexArray,
    sel_n1: IntArray,
    sel_m1: IntArray,
    sel_n2: IntArray,
    sel_m2: IntArray,
    max_d: int,
    max_order: int,
) -> ComplexArray:
    """Evaluate compressed exchange-kernel entries via the finite-q fast tables."""
    Nq = w_eff.size
    nG = kernels.shape[0]
    n_select = sel_n1.size

    shift_d = max_d
    shift_k = max_order

    out = np.empty((nG, n_select), dtype=np.complex128)

    for g in prange(nG):
        for j in range(n_select):
            n1 = sel_n1[j]
            m1 = sel_m1[j]
            n2 = sel_n2[j]
            m2 = sel_m2[j]

            din = m1 - n1
            dout = m2 - n2
            k = din - dout

            phase = phase_in[g, din + shift_d] * phase_out[g, dout + shift_d]
            ker_idx = k + shift_k

            acc = 0.0
            for iq in range(Nq):
                acc += (
                    w_eff[iq]
                    * kernels[g, iq, ker_idx]
                    * R[iq, n1, m1]
                    * R[iq, m2, n2]
                )

            out[g, j] = phase * acc

    return out


@_typed_njit(fastmath=False, parallel=True)
def _ogata_q_evaluate_numba(
    x_nodes: RealArray,
    w_eff: RealArray,
    k_og: RealArray,
    logfact: RealArray,
    sel_n1: IntArray,
    sel_m1: IntArray,
    sel_n2: IntArray,
    sel_m2: IntArray,
    phase_in: ComplexArray,
    phase_out: ComplexArray,
    max_d: int,
) -> ComplexArray:
    """Evaluate exchange kernels at Ogata-regime G values via q-space Ogata.

    Uses the Laguerre three-term recurrence for stable form-factor evaluation
    (no intermediate overflow), combined with Ogata quadrature weights that
    implicitly encode the Bessel oscillations.

    Parameters
    ----------
    x_nodes : (Nnodes,) Ogata x-nodes for a single Bessel order.
    w_eff : (nG_og, Nnodes) precomputed effective quadrature weights.
    k_og : (nG_og,) |G| magnitudes in the Ogata regime.
    logfact : (nmax+1,) log-factorials.
    sel_n1..sel_m2 : (nQ,) select-entry index arrays for this Bessel order.
    phase_in, phase_out : (nG_og, 2*max_d+1) phase tables.
    max_d : half-width of the d-index range.

    Returns
    -------
    out : (nG_og, nQ) complex128 array with phases applied.
    """
    nG = k_og.size
    Nnodes = x_nodes.size
    nQ = sel_n1.size
    nmax = logfact.size - 1
    shift_d = max_d

    out = np.zeros((nG, nQ), dtype=np.complex128)

    for g in prange(nG):
        k = k_og[g]
        Ltab = np.empty((nmax, nmax), dtype=np.float64)
        radial = np.zeros(nQ, dtype=np.float64)

        for iq in range(Nnodes):
            w = w_eff[g, iq]
            if abs(w) < 1e-300:
                continue

            q = x_nodes[iq] / k
            x = 0.5 * q * q

            # Three-term recurrence for L_kk^a(x)
            for a in range(nmax):
                Ltab[a, 0] = 1.0
                if nmax > 1:
                    Ltab[a, 1] = 1.0 + a - x
                for kk in range(1, nmax - 1):
                    Ltab[a, kk + 1] = (
                        (2 * kk + 1 + a - x) * Ltab[a, kk]
                        - (kk + a) * Ltab[a, kk - 1]
                    ) / (kk + 1)

            logx = np.log(x) if x > 0.0 else -700.0
            ex = np.exp(-0.5 * x)

            for j in range(nQ):
                # R_{n1,m1}(q)
                n1 = sel_n1[j]
                m1 = sel_m1[j]
                a1 = n1 - m1 if n1 >= m1 else m1 - n1
                k1 = n1 if n1 < m1 else m1
                if a1 == 0:
                    pxa1 = 1.0
                elif x > 0.0:
                    pxa1 = np.exp(0.5 * a1 * logx)
                else:
                    pxa1 = 0.0
                ratio1 = np.exp(0.5 * (logfact[k1] - logfact[k1 + a1]))
                r1 = ratio1 * pxa1 * Ltab[a1, k1] * ex

                # R_{m2,n2}(q)
                n2 = sel_n2[j]
                m2 = sel_m2[j]
                a2 = m2 - n2 if m2 >= n2 else n2 - m2
                k2 = m2 if m2 < n2 else n2
                if a2 == 0:
                    pxa2 = 1.0
                elif x > 0.0:
                    pxa2 = np.exp(0.5 * a2 * logx)
                else:
                    pxa2 = 0.0
                ratio2 = np.exp(0.5 * (logfact[k2] - logfact[k2 + a2]))
                r2 = ratio2 * pxa2 * Ltab[a2, k2] * ex

                radial[j] += w * r1 * r2

        # Apply phases and sign correction for negative Bessel order.
        # Ogata nodes are for J_{|k|}, but when k < 0: J_k = (-1)^|k| J_{|k|}.
        for j in range(nQ):
            din = sel_m1[j] - sel_n1[j]
            dout = sel_m2[j] - sel_n2[j]
            k_bessel = din - dout
            sign_corr = 1.0
            if k_bessel < 0 and (-k_bessel) % 2 == 1:
                sign_corr = -1.0
            phase = phase_in[g, din + shift_d] * phase_out[g, dout + shift_d]
            out[g, j] = sign_corr * radial[j] * phase

    return out


@dataclass(frozen=True)
class QuadratureParams:
    qmax: float
    N: int


@dataclass
class ExchangeFockPrecompute:
    nmax: int
    G_mags: RealArray
    G_thetas: RealArray
    sigma: float
    q_nodes: RealArray
    w_eff: RealArray
    R: RealArray
    kernels: RealArray
    phase_in: ComplexArray
    phase_out: ComplexArray
    max_d: int
    max_order: int
    include_minus: bool = True

    def exchange_fock(self, rho: ComplexArray) -> ComplexArray:
        rho = cast(ComplexArray, np.asarray(rho, dtype=np.complex128))
        if rho.shape != (self.G_mags.size, self.nmax, self.nmax):
            raise ValueError(
                f"rho shape must be (nG,nmax,nmax)=({self.G_mags.size},{self.nmax},{self.nmax})"
            )

        rho_c = np.ascontiguousarray(rho)
        F = _exchange_fock_numba(
            rho_c,
            self.R,
            self.w_eff,
            self.kernels,
            self.phase_in,
            self.phase_out,
            self.max_d,
            self.max_order,
        )

        return -F if self.include_minus else F


def build_exchange_fock_precompute(
    nmax: int,
    G_mags: RealArray,
    G_thetas: RealArray,
    params: QuadratureParams,
    *,
    sigma: float = +1.0,
    kappa: float = 1.0,
    potential: Callable[[RealArray], RealArray] | None = None,
    include_minus: bool = True,
) -> ExchangeFockPrecompute:
    G_mags = np.asarray(G_mags, dtype=np.float64)
    G_thetas = np.asarray(G_thetas, dtype=np.float64)
    if G_mags.ndim != 1 or G_thetas.ndim != 1 or G_mags.shape != G_thetas.shape:
        raise ValueError("G_mags and G_thetas must be 1D arrays with same shape")

    q_nodes, wq = _legendre_q_nodes_weights(params.N, params.qmax)

    if potential is None:
        w_eff = (float(kappa) * wq).astype(np.float64)
    else:
        Vraw = np.asarray(potential(q_nodes))
        if np.iscomplexobj(Vraw):
            raise ValueError("Callable potential must be real-valued.")
        V = Vraw.astype(np.float64, copy=False)
        w_eff = (wq * (q_nodes / (2.0 * np.pi)) * V).astype(np.float64)

    logfact = _logfact(int(nmax)).astype(np.float64)
    R = _precompute_R_table(q_nodes, logfact)

    max_d = int(nmax) - 1
    max_order = 2 * max_d
    kernels = _precompute_bessel_table(G_mags, q_nodes, max_order)

    phase_in, phase_out = _build_phase_tables(G_thetas, max_d, float(sigma))

    return ExchangeFockPrecompute(
        nmax=int(nmax),
        G_mags=G_mags,
        G_thetas=G_thetas,
        sigma=float(sigma),
        q_nodes=q_nodes,
        w_eff=w_eff,
        R=R,
        kernels=kernels,
        phase_in=phase_in,
        phase_out=phase_out,
        max_d=max_d,
        max_order=max_order,
        include_minus=bool(include_minus),
    )


def get_exchange_kernels_laguerre(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
    qmax: float = 35.0,
    nquad: int = 800,
    adaptive_nquad: bool = True,
    use_ogata: bool = False,
    ogata_h: float = 0.01,
    ogata_N: int | None = None,
    kmin_ogata: float = 20.0,
    sign_magneticfield: int = -1,
    select: Iterable[tuple[int, int, int, int]] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
) -> tuple[ComplexArray, list[tuple[int, int, int, int]]]:
    """Compute exchange kernels using the finite-q fast precompute tables.

    This backend uses Gauss-Legendre quadrature on ``[0, qmax]`` with a
    Numba-JIT form-factor table computed via the Laguerre three-term recurrence.
    This avoids intermediate overflow at large ``nmax`` and is numerically
    stable for arbitrarily high Landau-level indices.

    Parameters
    ----------
    adaptive_nquad : bool
        If True (default), automatically increase ``nquad`` so that the
        Gauss-Legendre grid resolves Bessel oscillations at the largest
        ``|G|`` in the input (at least 8 nodes per oscillation period).
    use_ogata : bool
        If True, use Ogata quadrature in q-space for ``|G| >= kmin_ogata``.
        This gives exponential convergence for oscillatory integrals with
        O(200) nodes regardless of ``|G|``, combined with the same stable
        R-table. Gauss-Legendre is still used for ``|G| < kmin_ogata``.
    ogata_h : float
        Ogata step-size parameter (smaller gives more nodes, higher accuracy).
    ogata_N : int or None
        Number of Ogata nodes per Bessel order. If None, uses ``int(pi/h)``.
    kmin_ogata : float
        Threshold on ``|G|`` above which Ogata quadrature is used (when
        ``use_ogata=True``).

    Returns
    -------
    values : ndarray, shape (nG, n_select), complex128
    select_list : list of (n1, m1, n2, m2) tuples
    """
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")

    G_magnitudes = np.asarray(G_magnitudes, dtype=np.float64).ravel()
    G_angles = np.asarray(G_angles, dtype=np.float64).ravel()
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")

    nmax = int(nmax)
    if nmax <= 0:
        raise ValueError("nmax must be positive")

    select_list, sel_n1, sel_m1, sel_n2, sel_m2 = normalize_select(
        nmax, select, canonical_select_max_entries=canonical_select_max_entries
    )

    # --- Adaptive nquad: ensure Bessel oscillations are resolved ---
    nquad_eff = int(nquad)
    qmax_f = float(qmax)
    if adaptive_nquad and G_magnitudes.size > 0:
        G_max = float(np.max(np.abs(G_magnitudes)))
        # 8 GL nodes per Bessel oscillation period (2*pi/G) on [0, qmax]
        nquad_bessel = int(np.ceil(8.0 * G_max * qmax_f / (2.0 * np.pi)))
        nquad_eff = max(nquad_eff, nquad_bessel)

    # --- Determine GL vs Ogata split ---
    og_mask = np.zeros(G_magnitudes.size, dtype=bool)
    if use_ogata and G_magnitudes.size > 0:
        og_mask = (G_magnitudes > 0.0) & (G_magnitudes >= float(kmin_ogata))

    gl_mask = ~og_mask
    og_idx = np.nonzero(og_mask)[0]
    gl_idx = np.nonzero(gl_mask)[0]

    # --- Potential kind ---
    is_callable = callable(potential)
    pot_kind = "" if is_callable else str(potential).strip().lower()
    if not is_callable and pot_kind not in ("coulomb", "constant"):
        raise ValueError("potential must be 'coulomb', 'constant', or a callable V(q)")

    max_d = nmax - 1
    logfact = _logfact(nmax).astype(np.float64)
    phase_in, phase_out = _build_phase_tables(
        G_angles, max_d, float(sign_magneticfield)
    )

    n_select = len(select_list)
    values: ComplexArray = np.zeros(
        (G_magnitudes.size, n_select), dtype=np.complex128
    )

    # --- GL path (for all G if no Ogata, or GL-regime subset) ---
    if gl_idx.size > 0 or not np.any(og_mask):
        # When Ogata is off, process all G; when on, process only GL subset
        if np.any(og_mask):
            G_gl = G_magnitudes[gl_idx]
            A_gl = G_angles[gl_idx]
            pi_gl, po_gl = _build_phase_tables(A_gl, max_d, float(sign_magneticfield))
        else:
            G_gl = G_magnitudes
            gl_idx = np.arange(G_magnitudes.size)
            pi_gl, po_gl = phase_in, phase_out

        # For GL-only path, adaptive nquad may be limited by kmin_ogata
        nquad_gl = nquad_eff
        if use_ogata and adaptive_nquad and G_gl.size > 0:
            G_max_gl = float(np.max(np.abs(G_gl)))
            nquad_gl = max(
                int(nquad),
                int(np.ceil(8.0 * G_max_gl * qmax_f / (2.0 * np.pi))),
            )
        q_nodes, wq = _legendre_q_nodes_weights(nquad_gl, qmax_f)

        if is_callable:
            assert callable(potential)
            Vraw = np.asarray(potential(q_nodes))
            if np.iscomplexobj(Vraw):
                raise ValueError("Callable potential must be real-valued.")
            V = Vraw.astype(np.float64, copy=False)
            if V.shape != q_nodes.shape:
                raise ValueError("Callable potential must return array of shape (nquad,)")
            w_eff = (wq * (q_nodes / (2.0 * np.pi)) * V).astype(np.float64)
        elif pot_kind == "coulomb":
            w_eff = (float(kappa) * wq).astype(np.float64)
        else:
            w_eff = (wq * (q_nodes / (2.0 * np.pi)) * float(kappa)).astype(np.float64)

        R = _precompute_R_table(q_nodes, logfact)

        k_abs_max = int(np.max(np.abs((sel_m1 - sel_n1) - (sel_m2 - sel_n2))))
        max_order = min(2 * max_d, k_abs_max)

        kernels = _precompute_bessel_table(G_gl, q_nodes, max_order)

        vals_gl = _evaluate_exchange_kernels_laguerre_numba(
                R,
                w_eff,
                kernels,
                pi_gl,
                po_gl,
                sel_n1.astype(np.int64, copy=False),
                sel_m1.astype(np.int64, copy=False),
                sel_n2.astype(np.int64, copy=False),
                sel_m2.astype(np.int64, copy=False),
                max_d,
                max_order,
            )
        values[gl_idx, :] = vals_gl

    # --- Ogata q-space path (for |G| >= kmin_ogata) ---
    if og_idx.size > 0:
        k_og = G_magnitudes[og_idx]
        pi_og = phase_in[og_idx]
        po_og = phase_out[og_idx]
        ogata_N_eff = int(ogata_N) if ogata_N is not None else int(np.pi / float(ogata_h))

        # Group select entries by absolute Bessel order
        k_arr = (sel_m1 - sel_n1) - (sel_m2 - sel_n2)
        abs_k_arr = np.abs(k_arr)
        unique_abs_k = np.unique(abs_k_arr)

        for abs_k_val in unique_abs_k:
            abs_k_val = int(abs_k_val)
            x_nodes, series_fac = _ogata_q_nodes(abs_k_val, float(ogata_h), ogata_N_eff)

            # Select entries with this Bessel order
            entry_mask = abs_k_arr == abs_k_val
            entry_idx = np.nonzero(entry_mask)[0]
            n1_g = sel_n1[entry_idx].astype(np.int64)
            m1_g = sel_m1[entry_idx].astype(np.int64)
            n2_g = sel_n2[entry_idx].astype(np.int64)
            m2_g = sel_m2[entry_idx].astype(np.int64)

            # Compute effective weights: w_eff_og[g, iq]
            # For ∫ f(q) J_ν(qk) dq ≈ Σ series_fac[n]/k * f(x[n]/k)
            sf = series_fac[None, :]  # (1, Nnodes)
            k_col = k_og[:, None]  # (nG_og, 1)

            if is_callable:
                assert callable(potential)
                q_2d = x_nodes[None, :] / k_col  # (nG_og, Nnodes)
                V_vals = np.asarray(potential(q_2d.ravel()), dtype=np.float64).reshape(
                    q_2d.shape
                )
                w_eff_og = (sf * V_vals * x_nodes[None, :] / (k_col**2 * 2.0 * np.pi)).astype(
                    np.float64
                )
            elif pot_kind == "coulomb":
                # Coulomb: 1/q cancels with q dq → weight = kappa * sf / k
                w_eff_og = (float(kappa) * sf / k_col).astype(np.float64)
            else:
                # Constant: V(q)=kappa, extra q/(2π) factor
                w_eff_og = (
                    float(kappa) * sf * x_nodes[None, :] / (k_col**2 * 2.0 * np.pi)
                ).astype(np.float64)

            vals_og = _ogata_q_evaluate_numba(
                    x_nodes,
                    w_eff_og,
                    k_og,
                    logfact,
                    n1_g,
                    m1_g,
                    n2_g,
                    m2_g,
                    pi_og,
                    po_og,
                    max_d,
                )
            values[np.ix_(og_idx, entry_idx)] = vals_og

    # Note: sign_magneticfield is already incorporated in the phase tables
    # built by _build_phase_tables(), so no post-hoc conjugation is needed.

    return values, select_list


__all__ = [
    "QuadratureParams",
    "ExchangeFockPrecompute",
    "build_exchange_fock_precompute",
    "cartesian_to_polar",
    "get_exchange_kernels_laguerre",
]
