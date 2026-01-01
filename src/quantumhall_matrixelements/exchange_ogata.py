"""Exchange kernels via Ogata quadrature (Hankel/Ogata) with robust fallbacks.

This module provides a high-performance drop-in alternative to the mapped
Gauss–Legendre implementation by using Ogata's quadrature for Hankel-type
integrals (as popularized by the `hankel` package), while retaining a mapped
Gauss–Legendre fallback for the small-|G| regime where Ogata can be inaccurate
(the x=k r change of variables concentrates support near x≈0).

The public entry point is :func:`get_exchange_kernels_Ogata`.
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.special as sps
from scipy.special import roots_legendre

from .diagnostic import get_exchange_kernels_opposite_field

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _parity_factor(N: int) -> int:
    """(-1)^((N-|N|)/2) → (-1)^N for N<0, and 1 for N>=0."""
    return int((-1) ** ((N - abs(N)) // 2))


@cache
def _legendre_nodes_weights_mapped(nquad: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, inf).

    Mapping: z = scale * (1+x)/(1-x)
    Jacobian: dz/dx = scale * 2/(1-x)^2
    """
    x, w_leg = roots_legendre(int(nquad))
    denom = 1.0 - x
    z = float(scale) * (1.0 + x) / denom
    w = w_leg * (float(scale) * 2.0 / (denom * denom))
    return np.asarray(z, dtype=np.float64), np.asarray(w, dtype=np.float64)


# -----------------------------------------------------------------------------
# Ogata quadrature helpers (bespoke, based on the same ingredients as `hankel`)
# -----------------------------------------------------------------------------
def _ogata_psi(t: np.ndarray) -> np.ndarray:
    """Ogata's double-exponential map ψ(t) = t * tanh((π/2) sinh t)."""
    return t * np.tanh(0.5 * np.pi * np.sinh(t))


def _ogata_dpsi(t: np.ndarray) -> np.ndarray:
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
def _ogata_nodes_weights(nu: int, h: float, N: int) -> tuple[np.ndarray, np.ndarray]:
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


def _broadcast_potential(V: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Broadcast a potential array to target_shape (nG, Nnodes) if possible.
    Accepts:
      - scalar
      - (Nnodes,)
      - (nG,)
      - (nG, Nnodes)
    """
    V = np.asarray(V)
    if V.ndim == 0:
        return np.broadcast_to(V, target_shape)
    if V.shape == target_shape:
        return V
    nG, Nn = target_shape
    if V.shape == (Nn,):
        return np.broadcast_to(V[None, :], target_shape)
    if V.shape == (nG,):
        return np.broadcast_to(V[:, None], target_shape)
    raise ValueError(
        f"Callable potential returned shape {V.shape}, expected broadcastable to {target_shape}"
    )


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------
def get_exchange_kernels_Ogata(
    G_magnitudes,
    G_angles,
    nmax: int,
    *,
    potential: str | Callable[[np.ndarray], np.ndarray] = "coulomb",
    kappa: float = 1.0,
    nquad: int = 8000,
    scale: float = 0.5,
    ell: float = 1.0,
    sign_magneticfield: int = -1,
    # Ogata-specific knobs
    ogata_h: float = 0.0075,
    ogata_N: int | None = None,
    kmin_ogata: float = 2.0,
    chunk_size: int = 128,
) -> "ComplexArray":
    """Compute exchange kernels using Ogata quadrature (Hankel/Ogata) with fallback.

    This function is a drop-in alternative to the mapped Gauss–Legendre method.
    For moderately large k = |G|ℓ it evaluates the oscillatory Bessel integrals
    using Ogata quadrature (exponentially convergent for Hankel transforms).
    For small k (including k=0), it automatically falls back to the original
    mapped Gauss–Legendre quadrature, where the integrand is non-oscillatory
    and Ogata can be inaccurate.

    Parameters
    ----------
    G_magnitudes, G_angles, nmax, potential, kappa, nquad, scale, ell, sign_magneticfield
        Same meaning as in the Gauss–Legendre implementation. ``nquad`` and
        ``scale`` are used for the fallback path.
    ogata_h : float
        Ogata step size h (smaller => more nodes => higher accuracy, slower).
        Default 0.0075.
    ogata_N : int or None
        Number of Ogata nodes. If None, uses int(pi/ogata_h).
    kmin_ogata : float
        Threshold on k=|G|ℓ below which we switch to the fallback quadrature.
        Default 2.0. If you know your regime, you can reduce it to use Ogata
        more aggressively.
    chunk_size : int
        Number of quadruples processed at once per N bucket (controls memory).
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
    if nmax <= 0:
        raise ValueError("nmax must be positive")

    ell = float(ell)
    if not (ell > 0.0):
        raise ValueError("ell must be > 0")

    # -----------------------------
    # 1. Potential choice
    # -----------------------------
    if callable(potential):
        pot_kind = "callable"
        pot_fn = potential
    else:
        pot_kind = str(potential).strip().lower()
        pot_fn = None

    if pot_kind not in ("coulomb", "callable"):
        raise ValueError("potential must be 'coulomb' or a callable V(q).")
    is_coulomb = pot_kind == "coulomb"

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
    buckets: dict[int, list[tuple[int, int, int, int]]] = {int(N): [] for N in Ns}

    for n1 in range(nmax):
        for m1 in range(nmax):
            D1 = int(D_nm[n1, m1])
            pair1 = n1 * nmax + m1
            for n2 in range(nmax):
                for m2 in range(nmax):
                    pair2 = m2 * nmax + n2  # partner indexing convention
                    if pair1 > pair2:
                        continue
                    D2 = int(D_nm[m2, n2])  # (m2 - n2)
                    N = int(D1 + D2)
                    buckets[N].append((n1, m1, n2, m2))

    # -----------------------------
    # 6. Split G-vectors into Ogata vs fallback groups
    # -----------------------------
    k_all = (G_magnitudes * ell).astype(np.float64)  # k = |G|ℓ
    # Ogata is used only for k>0 and above threshold.
    og_mask = (k_all > 0.0) & (k_all >= float(kmin_ogata))
    fb_mask = ~og_mask

    og_idx = np.nonzero(og_mask)[0]
    fb_idx = np.nonzero(fb_mask)[0]

    k_og = k_all[og_idx]
    k_fb = k_all[fb_idx]

    # -----------------------------
    # 7. Fallback quadrature precompute (only if needed)
    # -----------------------------
    fb_has = fb_idx.size > 0
    if fb_has:
        z_fb, w_fb = _legendre_nodes_weights_mapped(int(nquad), float(scale))
        exp_minus_z_fb = np.exp(-z_fb)
        sqrt2z_fb = np.sqrt(2.0 * z_fb)
        arg_fb = k_fb[:, None] * sqrt2z_fb[None, :]  # (nG_fb, nquad)

        # Callable potential: evaluated once on the fallback grid
        if not is_coulomb:
            qvals = sqrt2z_fb / ell  # (nquad,)
            Veff_fb = pot_fn(qvals) / (2.0 * np.pi * ell**2)
            Veff_fb = np.asarray(Veff_fb)
            if Veff_fb.shape != z_fb.shape:
                raise ValueError("Callable potential must return array of shape (nquad,) on fallback grid.")
        else:
            Veff_fb = None

        # Precompute z^alpha for all ds for fallback
        z_pows_fb = np.empty((max_d_sum + 1, z_fb.size), dtype=np.float64)
        if is_coulomb:
            # alpha = (ds - 1)/2
            for ds in range(max_d_sum + 1):
                alpha = 0.5 * (ds - 1)
                z_pows_fb[ds] = z_fb**alpha
        else:
            for ds in range(max_d_sum + 1):
                alpha = 0.5 * ds
                z_pows_fb[ds] = z_fb**alpha

        # Precompute Laguerre table on fallback grid for all keys
        # Shape: (nquad, nkeys_total)
        L_fb = np.empty((z_fb.size, nkeys_total), dtype=np.float64)
        for key in range(nkeys_total):
            p = int(key_p[key])
            d = int(key_d[key])
            L_fb[:, key] = sps.eval_genlaguerre(p, d, z_fb)

        # Cache of J_abs*w for each absN on fallback grid
        Jw_fb_cache: dict[int, np.ndarray] = {}
    else:
        # Placeholders
        z_fb = w_fb = exp_minus_z_fb = sqrt2z_fb = arg_fb = None
        Veff_fb = None
        z_pows_fb = None
        L_fb = None
        Jw_fb_cache = {}

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
    Xs = np.zeros((nG, nmax, nmax, nmax, nmax), dtype=np.complex128)
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
            quad_union.extend(buckets.get(int(N), []))
        if not quad_union:
            continue

        # -------------------------
        # Ogata precompute for this absN (only if needed)
        # -------------------------
        if og_has:
            x_nodes, wgt_nodes = _ogata_nodes_weights(absN, h, Nnodes)  # (Nnodes,), (Nnodes,)
            # Build u=z grid for all Ogata G-vectors
            k = k_og  # (nG_og,)
            # weights including 1/k^2
            W = (wgt_nodes[None, :] / (k[:, None] ** 2)).astype(np.float64)  # (nG_og, Nnodes)

            u = x_nodes[None, :] / k[:, None]  # (nG_og, Nnodes)
            u2 = u * u
            z = 0.5 * u2
            log2 = np.log(2.0)
            # log(u) - 0.5 log 2
            logu_minus = np.log(u) - 0.5 * log2
            common = -0.5 * u2

            # Potential on Ogata grid (if needed)
            if not is_coulomb:
                q = u / ell  # (nG_og, Nnodes)
                # Many user-supplied potentials are only written for 1D arrays.
                # Call on a flattened view and reshape back for robustness.
                Vraw = pot_fn(q.reshape(-1))
                Vraw = np.asarray(Vraw).reshape(q.shape)
                Veff_og = Vraw / (2.0 * np.pi * ell**2)
                Veff_og = _broadcast_potential(Veff_og, q.shape)
                pot_complex = np.iscomplexobj(Veff_og)
            else:
                Veff_og = None
                pot_complex = False

            # Determine all keys and ds values needed (union across +/- N)
            # Convert union list to arrays
            n1_u = np.fromiter((q[0] for q in quad_union), dtype=int, count=len(quad_union))
            m1_u = np.fromiter((q[1] for q in quad_union), dtype=int, count=len(quad_union))
            n2_u = np.fromiter((q[2] for q in quad_union), dtype=int, count=len(quad_union))
            m2_u = np.fromiter((q[3] for q in quad_union), dtype=int, count=len(quad_union))

            key1_g_u = key_nm[n1_u, m1_u]
            key2_g_u = key_nm[m2_u, n2_u]
            keys_needed = np.unique(np.concatenate([key1_g_u, key2_g_u]).astype(int))
            ds_u = d_nm[n1_u, m1_u] + d_nm[m2_u, n2_u]
            ds_needed = np.unique(ds_u.astype(int))

            # Local mapping from global key -> [0..nkeys-1] for this absN
            global_to_local = np.full(nkeys_total, -1, dtype=int)
            global_to_local[keys_needed] = np.arange(keys_needed.size, dtype=int)

            # Precompute Laguerre values on Ogata z grid for needed keys
            L_og = np.empty((k.size, x_nodes.size, keys_needed.size), dtype=np.float64)
            for j, key in enumerate(keys_needed):
                p = int(key_p[key])
                d = int(key_d[key])
                L_og[:, :, j] = sps.eval_genlaguerre(p, d, z)

            # Precompute WB_ds = W * exp(-u^2/2) * (u^2/2)^alpha * (Veff if any)
            WB_ds: list[np.ndarray | None] = [None] * (max_d_sum + 1)
            for ds in ds_needed:
                ds = int(ds)
                power = (ds - 1) if is_coulomb else ds
                # exponent = -u^2/2 + power * (log u - 0.5 log 2)
                base_real = np.exp(common + float(power) * logu_minus)
                if is_coulomb:
                    WB = base_real * W
                else:
                    WB = (base_real * Veff_og) * W  # may be complex
                WB_ds[ds] = WB

            # dtype for radial values on ogata path
            radial_dtype_og = np.complex128 if pot_complex else np.float64
        else:
            # placeholders to keep type-checkers happy
            x_nodes = wgt_nodes = None
            L_og = None
            WB_ds = None
            global_to_local = None
            keys_needed = None
            radial_dtype_og = np.float64
            pot_complex = False

        # -------------------------
        # Fallback Bessel cache for this absN (only if needed)
        # -------------------------
        if fb_has:
            if absN not in Jw_fb_cache:
                # J_abs(arg) on fallback subset, then multiply by w
                J_abs = sps.jv(absN, arg_fb)  # (nG_fb, nquad)
                Jw_fb_cache[absN] = (J_abs * w_fb[None, :]).astype(np.float64)
            Jw_fb = Jw_fb_cache[absN]
            radial_dtype_fb = np.complex128 if (not is_coulomb and np.iscomplexobj(Veff_fb)) else np.float64
        else:
            Jw_fb = None
            radial_dtype_fb = np.float64

        # Decide radial dtype for merged (ogata+fallback) block
        radial_dtype = np.complex128 if (radial_dtype_og == np.complex128 or radial_dtype_fb == np.complex128) else np.float64

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
            n1 = np.fromiter((q[0] for q in quad_list), dtype=int, count=nQ)
            m1 = np.fromiter((q[1] for q in quad_list), dtype=int, count=nQ)
            n2 = np.fromiter((q[2] for q in quad_list), dtype=int, count=nQ)
            m2 = np.fromiter((q[3] for q in quad_list), dtype=int, count=nQ)

            d1 = d_nm[n1, m1].astype(int)
            d2 = d_nm[m2, n2].astype(int)
            ds = (d1 + d2).astype(int)

            # Scalar prefactors (independent of G and quadrature)
            C1 = C_nm[n1, m1]
            C2 = C_nm[m2, n2]
            if is_coulomb:
                pref = (float(kappa) * C1 * C2 / sqrt2) * phase_power[ds]
            else:
                pref = (C1 * C2) * phase_power[ds]

            extra_sgns = extra_sign_nm[n2, m2].astype(np.int8)
            scalar = (pref * extra_sgns.astype(np.float64) * float(signN)).astype(np.complex128)  # (nQ,)

            # Precompute key indices for this bucket (global key indices always available)
            key1_g = key_nm[n1, m1].astype(int)          # (nQ,)
            key2_g = key_nm[m2, n2].astype(int)          # (nQ,)

            # For Ogata, map to local key indices within L_og
            if og_has:
                key1_l = global_to_local[key1_g]  # (nQ,)
                key2_l = global_to_local[key2_g]  # (nQ,)

            # Process in chunks for memory stability
            for start in range(0, nQ, int(chunk_size)):
                end = min(start + int(chunk_size), nQ)
                sl = slice(start, end)
                b = end - start

                n1_b = n1[sl]
                m1_b = m1[sl]
                n2_b = n2[sl]
                m2_b = m2[sl]
                ds_b = ds[sl]
                scalar_b = scalar[sl]

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
                        WB = WB_ds[ds_val]
                        if WB is None:
                            # This ds wasn't precomputed for this absN (shouldn't happen, but be safe)
                            power = (ds_val - 1) if is_coulomb else ds_val
                            base_real = np.exp(common + float(power) * logu_minus)
                            if is_coulomb:
                                WB = base_real * W
                            else:
                                WB = (base_real * Veff_og) * W
                            WB_ds[ds_val] = WB

                        mask = (ds_b == ds_val)
                        if not np.any(mask):
                            continue

                        # Gather Laguerres: (nG_og, Nnodes, bg)
                        L1 = L_og[:, :, key1l_b[mask]]
                        L2 = L_og[:, :, key2l_b[mask]]

                        # Weighted sum over nodes
                        radial_og[:, mask] = np.sum(WB[:, :, None] * L1 * L2, axis=1)

                    radial_block[og_idx, :] = radial_og

                # -------- Fallback contribution --------
                if fb_has:
                    # Build term matrix on fallback z grid for this block: (nquad, b)
                    # base_mat = exp(-z) * z^alpha * (Veff if any)
                    # We use z_pows_fb[ds_b] -> (b, nquad) then transpose.
                    zpow_sel = z_pows_fb[ds_b]  # (b, nquad)
                    base_mat = exp_minus_z_fb[:, None] * zpow_sel.T  # (nquad, b)
                    if not is_coulomb:
                        base_mat = base_mat * Veff_fb[:, None]

                    L1_fb = L_fb[:, key1_g[sl]]  # (nquad, b)
                    L2_fb = L_fb[:, key2_g[sl]]  # (nquad, b)
                    term_mat = base_mat * L1_fb * L2_fb  # (nquad, b)

                    radial_fb = Jw_fb @ term_mat  # (nG_fb, b), real or complex
                    radial_block[fb_idx, :] = radial_fb

                # Multiply in the N-dependent plane-wave phase and scalar prefactor.
                val_block = (phase_N[:, None] * radial_block) * scalar_b[None, :]  # (nG, b)

                # Scatter into output with advanced indexing
                Xs[:, n1_b, m1_b, n2_b, m2_b] = val_block

                # Fill symmetric partners (m2,n2,m1,n1) if pair1<pair2
                pair1_b = n1_b * nmax + m1_b
                pair2_b = m2_b * nmax + n2_b
                mask_partner = pair1_b < pair2_b
                if np.any(mask_partner):
                    delta = (n1_b - m1_b) - (n2_b - m2_b)
                    sign = np.where((delta & 1) == 0, 1, -1).astype(np.int8)
                    Xs[:, m2_b[mask_partner], n2_b[mask_partner], m1_b[mask_partner], n1_b[mask_partner]] = (
                        sign[mask_partner][None, :] * val_block[:, mask_partner]
                    )

    if sign_magneticfield == 1:
        Xs = get_exchange_kernels_opposite_field(Xs)

    return Xs


__all__ = ["get_exchange_kernels_Ogata"]
