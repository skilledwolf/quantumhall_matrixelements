"""Exchange kernels via Ogata quadrature (Hankel/Ogata) with robust fallbacks.

This module provides a high-performance drop-in alternative to the mapped
Gauss–Legendre implementation by using Ogata's quadrature for Hankel-type
integrals (as popularized by the ``hankel`` package), while retaining a mapped
Gauss–Legendre fallback for the small-``|G|`` regime where Ogata can be inaccurate
(the x=k r change of variables concentrates support near x≈0).

The public entry point is :func:`get_exchange_kernels_Ogata`.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache
from typing import cast

import numpy as np
import scipy.special as sps
from numpy.typing import NDArray

from ._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES, normalize_select

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Int8Array = NDArray[np.int8]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _parity_factor(N: int) -> int:
    """(-1)^((N-|N|)/2) → (-1)^N for N<0, and 1 for N>=0."""
    return int((-1) ** ((N - abs(N)) // 2))


# -----------------------------------------------------------------------------
# Ogata quadrature helpers (bespoke, based on the same ingredients as `hankel`)
# -----------------------------------------------------------------------------
def _ogata_psi(t: RealArray) -> RealArray:
    """Ogata's double-exponential map ψ(t) = t * tanh((π/2) sinh t)."""
    return np.asarray(t * np.tanh(0.5 * np.pi * np.sinh(t)), dtype=np.float64)


def _ogata_dpsi(t: RealArray) -> RealArray:
    """Derivative of ψ(t), with stabilization for large t."""
    t = np.asarray(t, dtype=np.float64)
    out = np.ones_like(t)
    # Past ~6, cosh(pi*sinh(t)) can overflow; in hankel it's effectively ~1.
    m = t < 6.0
    if np.any(m):
        tt = t[m]
        # (π t cosh t + sinh(π sinh t)) / (1 + cosh(π sinh t))
        out[m] = (np.pi * tt * np.cosh(tt) + np.sinh(np.pi * np.sinh(tt))) / (
            1.0 + np.cosh(np.pi * np.sinh(tt))
        )
    return out


@cache
def _ogata_nodes_weights(nu: int, h: float, N: int) -> tuple[RealArray, RealArray]:
    """
    Precompute Ogata nodes and weights for the Hankel transform of order `nu`.

    Returns
    -------
    x : (N,) float64
        Ogata nodes in the x variable (after mapping).
    wgt : (N,) float64
        Weights such that for the standard Hankel transform
            F(k) = ∫_0^∞ f(r) J_nu(k r) r dr
        we approximate
            F(k) ≈ Σ_n wgt[n] * f(x[n]/k) / k^2
        for k>0.
    """
    nu = int(nu)
    if nu < 0:
        raise ValueError("nu must be nonnegative")
    h = float(h)
    N = int(N)
    if not (h > 0.0):
        raise ValueError("ogata_h must be > 0")
    if N <= 0:
        raise ValueError("ogata_N must be a positive integer")

    # Bessel zeros j_{nu,n} / π
    roots = sps.jn_zeros(nu, N) / np.pi  # shape (N,)
    t = h * roots
    x = np.pi * _ogata_psi(t) / h

    # w_n = Y_nu(j_{nu,n}) / J_{nu+1}(j_{nu,n})
    j = np.pi * roots
    w = sps.yv(nu, j) / sps.jv(nu + 1, j)

    kernel = sps.jv(nu, x)  # non-alt kernel
    series_fac = np.pi * w * kernel * _ogata_dpsi(t)

    # For the standard transform, r_power=1 => multiply by x, divide by k^2.
    wgt = series_fac * x
    return np.asarray(x, dtype=np.float64), np.asarray(wgt, dtype=np.float64)


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------
def get_exchange_kernels_Ogata(
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
    # Ogata-specific knobs
    ogata_h: float = 0.0075,
    ogata_N: int | None = None,
    kmin_ogata: float = 2.0,
    chunk_size: int = 128,
    select: Iterable[tuple[int, int, int, int]] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
    ogata_auto: bool = False,
    ogata_auto_rtol: float = 3e-3,
    ogata_auto_atol: float = 1e-6,
    ogata_auto_refine: float = 2.0,
    ogata_auto_max_refine: int = 1,
    ogata_auto_fallback: str = "gausslegendre",
) -> tuple[ComplexArray, list[tuple[int, int, int, int]]]:
    r"""Compute exchange kernels using Ogata quadrature (Hankel/Ogata) with fallback.

    This function is a drop-in alternative to the mapped Gauss–Legendre method.
    For moderately large :math:`k = |G| \ell_B` it evaluates the oscillatory
    Bessel integrals
    using Ogata quadrature (exponentially convergent for Hankel transforms).
    For small k (including k=0), it automatically falls back to the original
    mapped Gauss–Legendre quadrature, where the integrand is non-oscillatory
    and Ogata can be inaccurate.

    Parameters
    ----------
    G_magnitudes, G_angles, nmax, potential, kappa, nquad, scale :
        Same meaning as in the Gauss–Legendre implementation. ``nquad`` and ``scale`` are
        used for the fallback path.
    ogata_h : float
        Ogata step size h (smaller => more nodes => higher accuracy, slower).
        Default 0.0075.
    ogata_N : int or None
        Number of Ogata nodes. If None, uses int(pi/ogata_h).
    kmin_ogata : float
        Threshold on :math:`k = |G| \ell_B` below which we switch to the fallback
        quadrature.
        Default 2.0. If you know your regime, you can reduce it to use Ogata
        more aggressively.
    chunk_size : int
        Number of quadruples processed at once per N bucket (controls memory).
    select : iterable of (n1, m1, n2, m2), optional
        If provided, compute only these entries and return an array of shape
        ``(nG, n_select)`` in the input order. This avoids allocating the full
        ``(nG, nmax, nmax, nmax, nmax)`` tensor.
    ogata_auto : bool, optional
        If True, attempt Ogata convergence by refining ``ogata_h`` (and
        ``ogata_N``) and fall back for any ``|G|`` that does not converge within
        tolerances.
    ogata_auto_rtol, ogata_auto_atol : float, optional
        Relative/absolute tolerances for the per-``|G|`` convergence check
        between successive Ogata refinements.
    ogata_auto_refine : float, optional
        Refinement factor for Ogata step size ``h`` (h -> h / refine).
    ogata_auto_max_refine : int, optional
        Maximum number of refinements. Must be >= 1 if ogata_auto is enabled.
    ogata_auto_fallback : {"gausslegendre", "hankel"}, optional
        Backend used for ``|G|`` values that fail Ogata convergence.

    Returns
    -------
    values : numpy.ndarray (nG, n_select)
        Compressed exchange values matching ``select_list``.
    select_list : list[tuple[int, int, int, int]]
        Quadruples corresponding to the columns of ``values``.
    """
    # -----------------------------
    # 0. Input handling
    # -----------------------------
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")

    G_magnitudes = np.asarray(G_magnitudes, dtype=float).ravel()
    G_angles = np.asarray(G_angles, dtype=float).ravel()
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")
    nG = int(G_magnitudes.size)

    nmax = int(nmax)

    select_list, sel_n1, sel_m1, sel_n2, sel_m2 = normalize_select(
        nmax, select, canonical_select_max_entries=canonical_select_max_entries
    )

    if ogata_auto:
        if ogata_auto_max_refine < 1:
            raise ValueError("ogata_auto_max_refine must be >= 1 when ogata_auto is enabled.")
        if ogata_auto_refine <= 1.0:
            raise ValueError("ogata_auto_refine must be > 1 when ogata_auto is enabled.")

        def _per_g_max_abs(arr: ComplexArray) -> RealArray:
            return cast(RealArray, np.max(np.abs(arr), axis=1))

        k_all = np.asarray(G_magnitudes, dtype=float)
        og_mask = (k_all > 0.0) & (k_all >= float(kmin_ogata))
        if not np.any(og_mask):
            return get_exchange_kernels_Ogata(
                G_magnitudes,
                G_angles,
                nmax,
                potential=potential,
                kappa=kappa,
                nquad=nquad,
                scale=scale,
                nlag=nlag,
                sign_magneticfield=sign_magneticfield,
                ogata_h=ogata_h,
                ogata_N=ogata_N,
                kmin_ogata=kmin_ogata,
                chunk_size=chunk_size,
                select=select,
                canonical_select_max_entries=canonical_select_max_entries,
                ogata_auto=False,
            )

        h_curr = float(ogata_h)
        N_curr = ogata_N
        with np.errstate(over="ignore", invalid="ignore"):
            vals_prev, sel_prev = get_exchange_kernels_Ogata(
                G_magnitudes,
                G_angles,
                nmax,
                potential=potential,
                kappa=kappa,
                nquad=nquad,
                scale=scale,
                nlag=nlag,
                sign_magneticfield=sign_magneticfield,
                ogata_h=h_curr,
                ogata_N=N_curr,
                kmin_ogata=kmin_ogata,
                chunk_size=chunk_size,
                select=select,
                ogata_auto=False,
            )
        result_select = sel_prev

        ok_mask = np.zeros_like(og_mask, dtype=bool)
        vals_curr = vals_prev
        for _ in range(int(ogata_auto_max_refine)):
            h_curr = h_curr / float(ogata_auto_refine)
            if N_curr is None:
                N_next = None
            else:
                N_next = int(np.ceil(int(N_curr) * float(ogata_auto_refine)))

            with np.errstate(over="ignore", invalid="ignore"):
                vals_curr, sel_curr = get_exchange_kernels_Ogata(
                    G_magnitudes,
                    G_angles,
                    nmax,
                    potential=potential,
                    kappa=kappa,
                    nquad=nquad,
                    scale=scale,
                    nlag=nlag,
                    sign_magneticfield=sign_magneticfield,
                    ogata_h=h_curr,
                    ogata_N=N_next,
                    kmin_ogata=kmin_ogata,
                    chunk_size=chunk_size,
                    select=select,
                    ogata_auto=False,
                )
            if sel_curr != result_select:
                raise RuntimeError("select list changed between Ogata auto refinements")

            diff = _per_g_max_abs(vals_curr - vals_prev)
            scale_ref = np.maximum(_per_g_max_abs(vals_curr), _per_g_max_abs(vals_prev))
            tol = np.maximum(float(ogata_auto_atol), float(ogata_auto_rtol) * scale_ref)
            ok = (diff <= tol) & np.isfinite(diff) & np.isfinite(scale_ref)
            ok_mask = (~og_mask) | ok
            if np.all(ok_mask):
                break
            vals_prev = vals_curr
            N_curr = N_next

        result_vals = vals_curr
        bad_mask = og_mask & ~ok_mask
        if np.any(bad_mask):
            fallback = str(ogata_auto_fallback).strip().lower()
            if fallback not in {"gausslegendre", "hankel"}:
                raise ValueError("ogata_auto_fallback must be 'gausslegendre' or 'hankel'.")
            G_bad = np.asarray(G_magnitudes)[bad_mask]
            A_bad = np.asarray(G_angles)[bad_mask]
            if fallback == "hankel":
                from .exchange_hankel import get_exchange_kernels_hankel

                X_bad, sel_bad = get_exchange_kernels_hankel(
                    G_bad,
                    A_bad,
                    nmax,
                    potential=potential,
                    kappa=kappa,
                    sign_magneticfield=sign_magneticfield,
                    select=select,
                )
            else:
                from .exchange_legendre import get_exchange_kernels_GaussLegendre

                X_bad, sel_bad = get_exchange_kernels_GaussLegendre(
                    G_bad,
                    A_bad,
                    nmax,
                    potential=potential,
                    kappa=kappa,
                    nquad=nquad,
                    scale=scale,
                    sign_magneticfield=sign_magneticfield,
                    select=select,
                )
            if sel_bad != result_select:
                raise RuntimeError("fallback backend returned different select list")
            result_vals = np.array(result_vals, copy=True)
            result_vals[bad_mask] = X_bad

        return result_vals, result_select
    if nmax <= 0:
        raise ValueError("nmax must be positive")
    # -----------------------------
    # 1. Potential choice
    # -----------------------------
    if callable(potential):
        pot_kind = "callable"
        pot_fn = potential
    else:
        pot_kind = str(potential).strip().lower()
        pot_fn = None

    if pot_kind not in ("coulomb", "constant", "callable"):
        raise ValueError("potential must be 'coulomb', 'constant', or a callable V(q).")
    is_coulomb = pot_kind == "coulomb"
    is_constant = pot_kind == "constant"

    # -----------------------------
    # 2. Derived indexing/combinatorics (vectorized)
    # -----------------------------
    idx = np.arange(nmax, dtype=int)
    n_idx, m_idx = np.meshgrid(idx, idx, indexing="ij")

    p_nm = np.minimum(n_idx, m_idx)
    d_nm = np.abs(n_idx - m_idx)
    D_nm = n_idx - m_idx

    # (-1)^(n-m) via parity bits, shape (nmax, nmax)
    extra_sign_nm = 1 - 2 * ((n_idx - m_idx) & 1)

    # C_nm[n,m] = sqrt(p! / (p + d)!)
    max_fact = 2 * (nmax - 1)
    logfact = sps.gammaln(np.arange(max_fact + 1, dtype=np.float64) + 1.0)  # log(n!)
    C_nm = np.exp(0.5 * (logfact[p_nm] - logfact[p_nm + d_nm])).astype(np.float64)

    # -----------------------------
    # 3. Key indexing for Laguerre L_p^d:
    #    global key index over all (p,d) pairs.
    # -----------------------------
    # For fixed d, p ranges 0..(nmax-d-1) => count nmax-d
    # offset_d[d] = sum_{j<d} (nmax-j) = d*nmax - d*(d-1)/2
    ar = np.arange(nmax, dtype=int)
    offset_d = (ar * nmax) - (ar * (ar - 1) // 2)
    nkeys_total = nmax * (nmax + 1) // 2

    # For each (n,m) pair, global key index:
    key_nm = offset_d[d_nm] + p_nm  # shape (nmax, nmax), int

    # Precompute key->(p,d) lookup arrays
    key_p = np.empty(nkeys_total, dtype=int)
    key_d = np.empty(nkeys_total, dtype=int)
    kk = 0
    for d in range(nmax):
        for p in range(nmax - d):
            key_p[kk] = p
            key_d[kk] = d
            kk += 1

    # -----------------------------
    # 4. N-related tables: parity, plane-wave phase, phase powers
    # -----------------------------
    maxD = 2 * (nmax - 1)
    Ns = np.arange(-maxD, maxD + 1, dtype=int)
    minN = int(Ns[0])

    parity = np.array([_parity_factor(int(N)) for N in Ns], dtype=np.int8)
    phase_table = np.exp(-1j * Ns[:, None] * G_angles[None, :])  # (2*maxD+1, nG)

    max_d_sum = 2 * (nmax - 1)
    # (1j)^(d1 + d2)
    phase_power = (1j) ** np.arange(max_d_sum + 1, dtype=int)

    # -----------------------------
    # 5. Build buckets of quadruples by N (same symmetry strategy)
    # -----------------------------
    buckets: dict[int, list[tuple[int, int, int, int, int]]] = {int(N): [] for N in Ns}
    for idx_sel, (n1, m1, n2, m2) in enumerate(select_list):
        D1 = int(D_nm[n1, m1])
        D2 = int(D_nm[m2, n2])  # (m2 - n2)
        N = int(D1 + D2)
        buckets[N].append((n1, m1, n2, m2, idx_sel))

    # -----------------------------
    # 6. Split G-vectors into Ogata vs fallback groups
    # -----------------------------
    k_all = np.asarray(G_magnitudes, dtype=float).astype(np.float64)  # k = |G|
    # Ogata is used only for k>0 and above threshold.
    og_mask = (k_all > 0.0) & (k_all >= float(kmin_ogata))
    fb_mask = ~og_mask

    og_idx = np.nonzero(og_mask)[0]
    fb_idx = np.nonzero(fb_mask)[0]

    k_og = k_all[og_idx]
    fb_values: ComplexArray | None = None
    if fb_idx.size > 0:
        from .exchange_legendre import get_exchange_kernels_GaussLegendre

        fb_values, fb_select = get_exchange_kernels_GaussLegendre(
            np.asarray(G_magnitudes)[fb_idx],
            np.asarray(G_angles)[fb_idx],
            nmax,
            potential=potential,
            kappa=kappa,
            nquad=nquad,
            scale=scale,
            nlag=nlag,
            sign_magneticfield=-1,
            select=select,
            canonical_select_max_entries=canonical_select_max_entries,
        )
        if fb_select != select_list:
            raise RuntimeError("fallback backend returned different select list")

    # -----------------------------
    # 8. Ogata quadrature setup (global)
    # -----------------------------
    og_has = og_idx.size > 0
    if og_has:
        h = float(ogata_h)
        Nnodes = int(ogata_N) if ogata_N is not None else int(np.pi / h)
        if Nnodes <= 0:
            raise ValueError("ogata_N must be positive (or ogata_h must be > 0).")
    else:
        h = float(ogata_h)
        Nnodes = 0

    # -----------------------------
    # 9. Output array
    # -----------------------------
    Xs: ComplexArray = np.zeros((nG, len(select_list)), dtype=np.complex128)
    sqrt2 = np.sqrt(2.0)

    # -----------------------------
    # 10. Main loop grouped by absN for reuse
    # -----------------------------
    for absN in range(maxD + 1):
        # Gather N buckets with this absN
        Ns_here: list[int] = [absN]
        if absN != 0:
            Ns_here.append(-absN)

        # Quick check: if no quadruples for either sign, skip
        quad_union: list[tuple[int, int, int, int]] = []
        for N in Ns_here:
            quad_items = buckets.get(int(N), [])
            if not quad_items:
                continue
            quad_union.extend((q[0], q[1], q[2], q[3]) for q in quad_items)
        if not quad_union:
            continue

        # -------------------------
        # Ogata precompute for this absN (only if needed)
        # -------------------------
        x_nodes: RealArray = np.empty((0,), dtype=np.float64)
        wgt_nodes: RealArray = np.empty((0,), dtype=np.float64)
        W_og: RealArray = np.empty((0, 0), dtype=np.float64)
        common_og: RealArray = np.empty((0, 0), dtype=np.float64)
        logu_minus_og: RealArray = np.empty((0, 0), dtype=np.float64)
        Veff_og: RealArray = np.empty((0, 0), dtype=np.float64)
        global_to_local: IntArray = np.full(nkeys_total, -1, dtype=np.int64)
        L_og: RealArray = np.empty((0, 0, 0), dtype=np.float64)
        WB_ds: list[RealArray | None] = [None] * (max_d_sum + 1)

        if og_has:
            x_nodes, wgt_nodes = _ogata_nodes_weights(absN, h, Nnodes)  # (Nnodes,), (Nnodes,)
            k = k_og  # (nG_og,)
            W_og = (wgt_nodes[None, :] / (k[:, None] ** 2)).astype(np.float64)  # (nG_og, Nnodes)

            u = x_nodes[None, :] / k[:, None]  # (nG_og, Nnodes)
            u2 = u * u
            z_og = 0.5 * u2
            logu_minus_og = np.log(u) - 0.5 * np.log(2.0)
            common_og = -0.5 * u2

            if not is_coulomb:
                if is_constant:
                    Veff_og = (float(kappa) / (2.0 * np.pi)) * np.ones_like(u)
                else:
                    assert pot_fn is not None
                    Vraw = np.asarray(pot_fn(u.reshape(-1)))
                    if np.iscomplexobj(Vraw):
                        raise ValueError("Callable potential must be real-valued.")
                    try:
                        Vraw = Vraw.reshape(u.shape)
                    except ValueError as exc:
                        raise ValueError(
                            "Callable potential must return an array compatible with the Ogata "
                            "grid shape."
                        ) from exc
                    Vraw = Vraw.astype(np.float64, copy=False)
                    Veff_og = Vraw / (2.0 * np.pi)

            # Determine all keys and ds values needed (union across +/- N)
            n1_u = np.fromiter((q[0] for q in quad_union), dtype=int, count=len(quad_union))
            m1_u = np.fromiter((q[1] for q in quad_union), dtype=int, count=len(quad_union))
            n2_u = np.fromiter((q[2] for q in quad_union), dtype=int, count=len(quad_union))
            m2_u = np.fromiter((q[3] for q in quad_union), dtype=int, count=len(quad_union))

            key1_g_u = key_nm[n1_u, m1_u]
            key2_g_u = key_nm[m2_u, n2_u]
            keys_needed = np.unique(np.concatenate([key1_g_u, key2_g_u]).astype(np.int64))
            ds_u = d_nm[n1_u, m1_u] + d_nm[m2_u, n2_u]
            ds_needed = np.unique(ds_u.astype(int))

            # Local mapping from global key -> [0..nkeys-1] for this absN
            global_to_local = np.full(nkeys_total, -1, dtype=np.int64)
            global_to_local[keys_needed] = np.arange(keys_needed.size, dtype=np.int64)

            # Precompute Laguerre values on Ogata z grid for needed keys
            L_og = np.empty((k.size, x_nodes.size, keys_needed.size), dtype=np.float64)
            for j, key in enumerate(keys_needed):
                p = int(key_p[key])
                d = int(key_d[key])
                L_og[:, :, j] = sps.eval_genlaguerre(p, d, z_og)

            # Precompute WB_ds = W * exp(-u^2/2) * (u^2/2)^alpha * (Veff if any)
            for ds_idx in ds_needed:
                ds_idx = int(ds_idx)
                power = (ds_idx - 1) if is_coulomb else ds_idx
                base_real = np.exp(common_og + float(power) * logu_minus_og)
                WB = (
                    base_real * W_og if is_coulomb else (base_real * Veff_og) * W_og
                )
                WB_ds[ds_idx] = WB

        radial_dtype = np.float64

        # -------------------------
        # Process each N bucket for this absN
        # -------------------------
        for N in Ns_here:
            quad_list = buckets.get(int(N), [])
            if not quad_list:
                continue

            # N-specific phase/sign
            N_idx = int(N - minN)
            signN = int(parity[N_idx])
            phase_N = phase_table[N_idx]  # (nG,)

            # Convert quads to vector arrays
            nQ = len(quad_list)
            n1_arr = np.fromiter((q[0] for q in quad_list), dtype=int, count=nQ)
            m1_arr = np.fromiter((q[1] for q in quad_list), dtype=int, count=nQ)
            n2_arr = np.fromiter((q[2] for q in quad_list), dtype=int, count=nQ)
            m2_arr = np.fromiter((q[3] for q in quad_list), dtype=int, count=nQ)
            sel_idx_arr = np.fromiter((q[4] for q in quad_list), dtype=int, count=nQ)

            d1 = d_nm[n1_arr, m1_arr].astype(int)
            d2 = d_nm[m2_arr, n2_arr].astype(int)
            ds_arr = (d1 + d2).astype(int)

            # Scalar prefactors (independent of G and quadrature)
            C1 = C_nm[n1_arr, m1_arr]
            C2 = C_nm[m2_arr, n2_arr]
            if is_coulomb:
                pref = (float(kappa) * C1 * C2 / sqrt2) * phase_power[ds_arr]
            else:
                pref = (C1 * C2) * phase_power[ds_arr]

            extra_sgns = extra_sign_nm[n2_arr, m2_arr].astype(np.int8)
            scalar = (pref * extra_sgns.astype(np.float64) * float(signN)).astype(
                np.complex128
            )  # (nQ,)

            # Precompute key indices for this bucket (global key indices always available)
            key1_g = key_nm[n1_arr, m1_arr].astype(int)  # (nQ,)
            key2_g = key_nm[m2_arr, n2_arr].astype(int)  # (nQ,)

            # For Ogata, map to local key indices within L_og
            if og_has:
                key1_l = global_to_local[key1_g]  # (nQ,)
                key2_l = global_to_local[key2_g]  # (nQ,)

            # Process in chunks for memory stability
            for start in range(0, nQ, int(chunk_size)):
                end = min(start + int(chunk_size), nQ)
                sl = slice(start, end)
                b = end - start

                ds_b = ds_arr[sl]
                scalar_b = scalar[sl]
                sel_idx_b = sel_idx_arr[sl]

                # radial_block: (nG, b)
                radial_block = np.zeros((nG, b), dtype=radial_dtype)

                # -------- Ogata contribution --------
                if og_has:
                    # Compute radial only for og_idx subset and fill
                    radial_og = np.empty((og_idx.size, b), dtype=radial_dtype)
                    radial_og.fill(0)

                    key1l_b = key1_l[sl]
                    key2l_b = key2_l[sl]

                    # Group by ds to reuse WB_ds[ds]
                    for ds_val in np.unique(ds_b):
                        ds_val = int(ds_val)
                        WB_opt = WB_ds[ds_val]
                        if WB_opt is None:
                            # This ds wasn't precomputed for this absN (shouldn't happen; be safe).
                            power = (ds_val - 1) if is_coulomb else ds_val
                            base_real = np.exp(common_og + float(power) * logu_minus_og)
                            WB = (
                                base_real * W_og
                                if is_coulomb
                                else (base_real * Veff_og) * W_og
                            )
                            WB_ds[ds_val] = WB
                        else:
                            WB = WB_opt

                        mask = (ds_b == ds_val)
                        if not np.any(mask):
                            continue

                        # Gather Laguerres: (nG_og, Nnodes, bg)
                        L1 = L_og[:, :, key1l_b[mask]]
                        L2 = L_og[:, :, key2l_b[mask]]

                        # Weighted sum over nodes
                        radial_og[:, mask] = np.sum(WB[:, :, None] * L1 * L2, axis=1)

                    radial_block[og_idx, :] = radial_og

                # Multiply in the N-dependent plane-wave phase and scalar prefactor.
                val_block = (phase_N[:, None] * radial_block) * scalar_b[None, :]  # (nG, b)

                # Scatter into output
                Xs[:, sel_idx_b] = val_block

    if fb_values is not None:
        Xs[fb_idx] = fb_values

    if sign_magneticfield == 1:
        phase1 = 1 - 2 * ((sel_n1 - sel_m1) & 1)
        phase2 = 1 - 2 * ((sel_n2 - sel_m2) & 1)
        Xs = Xs.conj() * (phase1 * phase2)[None, :]

    return Xs, select_list


__all__ = ["get_exchange_kernels_Ogata"]
