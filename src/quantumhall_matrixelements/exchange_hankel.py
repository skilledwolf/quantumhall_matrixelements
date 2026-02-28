"""Exchange kernels via Hankel transforms (vectorized)."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import cast

import numpy as np
import scipy.special as sps
from hankel import HankelTransform
from numpy.typing import NDArray

from ._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES, normalize_select

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def _parity_factor(N: int) -> int:
    return int((-1) ** ((N - abs(N)) // 2))


@cache
def _get_hankel_nodes(nu: int, N: int, h: float) -> tuple[RealArray, RealArray, int, int, float]:
    ht = HankelTransform(nu=int(nu), N=int(N), h=float(h))
    x = np.asarray(ht.x, dtype=np.float64)
    try:
        series_fac = np.asarray(ht._series_fac, dtype=np.float64)
        r_power = int(ht._r_power)
        k_power = int(ht._k_power)
        norm = float(ht._norm(False))
    except AttributeError as exc:  # pragma: no cover - depends on external package internals
        try:
            hankel_ver = _metadata_version("hankel")
        except PackageNotFoundError:
            hankel_ver = "unknown"
        raise RuntimeError(
            "Incompatible 'hankel' package detected while extracting Hankel quadrature nodes. "
            "This backend currently relies on internal HankelTransform attributes "
            "('_series_fac', '_r_power', '_k_power', '_norm') that may change between versions. "
            f"Detected hankel=={hankel_ver}. Please install a compatible version."
        ) from exc
    return x, series_fac, r_power, k_power, norm


def get_exchange_kernels_hankel(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
    sign_magneticfield: int = -1,
    select: Iterable[tuple[int, int, int, int]] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
    hankel_N: int = 6000,
    hankel_h: float = 7e-6,
    chunk_size: int = 128,
    hankel_nlag: int = 80,
    hankel_q_cut: float | None = 7.5,
    hankel_trunc_kmin: float = 2.0,
) -> tuple[ComplexArray, list[tuple[int, int, int, int]]]:
    """Compute X_{n1,m1,n2,m2}(G) via vectorized Hankel quadrature.

    Parameters
    ----------
    G_magnitudes, G_angles :
        Arrays describing |G| and polar angle θ_G (same shape, no broadcasting).
    nmax :
        Number of Landau levels.
    potential :
        ``'coulomb'`` (default), ``'constant'``, or a callable ``V(q)`` giving
        the interaction in 1/ℓ units.
    kappa :
        Prefactor for Coulomb/constant cases.
    sign_magneticfield :
        Sign of the charge–field product σ = sgn(q B_z). ``-1`` matches the
        package's internal convention; ``+1`` returns the kernels for the
        opposite sign by applying the appropriate complex conjugation and
        phase factors.
    select :
        Optional iterable of (n1, m1, n2, m2). If provided, only these entries
        are computed and the return array has shape ``(nG, n_select)`` in the
        input order. If omitted, a canonical set of representatives is used and
        returned in compressed form to avoid materializing the full tensor.
    hankel_N, hankel_h :
        Hankel transform grid parameters (passed to ``hankel.HankelTransform``).
    chunk_size :
        Number of quadruples processed at once per N bucket.
    hankel_nlag :
        Order of generalized Gauss-Laguerre quadrature used for the k=0 slice.
    hankel_q_cut :
        Adaptive node window for k>0: keep nodes with q = x/k <= hankel_q_cut.
    hankel_trunc_kmin :
        Disable node truncation for k < hankel_trunc_kmin (keeps full grid for small k).

    Returns
    -------
    values : numpy.ndarray (nG, n_select)
        Compressed exchange values matching ``select_list``.
    select_list : list[tuple[int,int,int,int]]
        Quadruples corresponding to the columns of ``values``.
    """
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")

    G_magnitudes = np.asarray(G_magnitudes, dtype=float).ravel()
    G_angles = np.asarray(G_angles, dtype=float).ravel()
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have same shape")

    nG = int(G_magnitudes.size)
    nmax = int(nmax)

    select_list, sel_n1, sel_m1, sel_n2, sel_m2 = normalize_select(
        nmax, select, canonical_select_max_entries=canonical_select_max_entries
    )

    # Resolve potential
    if callable(potential):
        pot_kind = "callable"
        pot_fn = potential
    else:
        pot_kind = str(potential).strip().lower()
        pot_fn = None
    if pot_kind not in {"coulomb", "constant", "callable"}:
        raise ValueError("potential must be 'coulomb', 'constant', or callable V(q)")
    is_coulomb = pot_kind == "coulomb"
    is_constant = pot_kind == "constant"

    # -----------------------------
    # 1. Indexing/combinatorics
    # -----------------------------
    idx = np.arange(nmax, dtype=int)
    n_idx, m_idx = np.meshgrid(idx, idx, indexing="ij")

    p_nm = np.minimum(n_idx, m_idx)
    d_nm = np.abs(n_idx - m_idx)
    D_nm = n_idx - m_idx

    extra_sign_nm = 1 - 2 * ((n_idx - m_idx) & 1)

    # normalization terms
    d_vals = np.arange(nmax, dtype=int)
    nrm_d = np.exp(-0.5 * sps.gammaln(d_vals + 1.0))  # sqrt(rgamma(d+1))

    log_nrm_lag = 0.5 * (
        sps.gammaln(p_nm + d_nm + 1.0)
        - sps.gammaln(p_nm + 1.0)
        - sps.gammaln(d_nm + 1.0)
    )
    nrm_lag = np.exp(log_nrm_lag)

    # -----------------------------
    # 2. Key indexing for Laguerre L_p^d
    # -----------------------------
    ar = np.arange(nmax, dtype=int)
    offset_d = (ar * nmax) - (ar * (ar - 1) // 2)
    nkeys_total = nmax * (nmax + 1) // 2

    key_nm = offset_d[d_nm] + p_nm

    key_p = np.empty(nkeys_total, dtype=int)
    key_d = np.empty(nkeys_total, dtype=int)
    kk = 0
    for d in range(nmax):
        for p in range(nmax - d):
            key_p[kk] = p
            key_d[kk] = d
            kk += 1

    # -----------------------------
    # 3. N-related tables
    # -----------------------------
    maxD = 2 * (nmax - 1)
    Ns = np.arange(-maxD, maxD + 1, dtype=int)
    minN = int(Ns[0])

    parity = np.array([_parity_factor(int(N)) for N in Ns], dtype=np.int8)
    phase_table = np.exp(-1j * Ns[:, None] * G_angles[None, :])

    max_d_sum = 2 * (nmax - 1)
    phase_power = (1j) ** np.arange(max_d_sum + 1, dtype=int)

    # -----------------------------
    # 4. Build buckets by N (canonical pairs)
    # -----------------------------
    buckets: dict[int, list[tuple[int, int, int, int, int]]] = {int(N): [] for N in Ns}
    for idx_sel, (n1, m1, n2, m2) in enumerate(select_list):
        D1 = int(D_nm[n1, m1])
        D2 = int(D_nm[m2, n2])
        N = int(D1 + D2)
        buckets[N].append((n1, m1, n2, m2, idx_sel))

    # -----------------------------
    # 5. Split k=0 vs k>0
    # -----------------------------
    k_all = G_magnitudes.astype(np.float64)
    nz_mask = k_all > 0.0
    nz_idx = np.nonzero(nz_mask)[0]
    z_idx = np.nonzero(~nz_mask)[0]

    Xs: ComplexArray = np.zeros((nG, len(select_list)), dtype=np.complex128)

    k_nz = k_all[nz_idx]
    sqrt2 = np.sqrt(2.0)

    # -----------------------------
    # 6. Main loop grouped by absN
    # -----------------------------
    if k_nz.size > 0:
        for absN in range(maxD + 1):
            Ns_here = [absN]
            if absN != 0:
                Ns_here.append(-absN)

            quad_union: list[tuple[int, int, int, int]] = []
            for N in Ns_here:
                quad_items = buckets.get(int(N), [])
                if not quad_items:
                    continue
                quad_union.extend((q[0], q[1], q[2], q[3]) for q in quad_items)
            if not quad_union:
                continue

            x, series_fac, r_power, k_power, norm = _get_hankel_nodes(absN, hankel_N, hankel_h)
            x_pow = x**r_power
            denom_pow = k_power + r_power + 1

            # ds values that can appear for this |N|
            n1_u = np.fromiter((q[0] for q in quad_union), dtype=int, count=len(quad_union))
            m1_u = np.fromiter((q[1] for q in quad_union), dtype=int, count=len(quad_union))
            n2_u = np.fromiter((q[2] for q in quad_union), dtype=int, count=len(quad_union))
            m2_u = np.fromiter((q[3] for q in quad_union), dtype=int, count=len(quad_union))
            ds_needed = np.unique(d_nm[n1_u, m1_u] + d_nm[m2_u, n2_u]).astype(int)

            # Adaptive truncation per k: keep nodes with q <= hankel_q_cut.
            # For small k (below hankel_trunc_kmin) we disable truncation to
            # avoid accuracy loss in the low-|G| regime.
            if hankel_q_cut is None:
                idx_cut = np.full(k_nz.shape, x.size, dtype=int)
            else:
                qcut = float(hankel_q_cut)
                idx_cut = np.searchsorted(x, k_nz * qcut, side="right")
                # keep at least as large as qcut but also allow a hard cap on max nodes
                if hankel_trunc_kmin is not None:
                    small_mask = k_nz < float(hankel_trunc_kmin)
                    if np.any(small_mask):
                        idx_cut = np.where(small_mask, x.size, idx_cut)
                idx_cut = np.maximum(idx_cut, 1)

            unique_lengths = np.unique(idx_cut)

            for m_nodes in unique_lengths:
                m_nodes = int(m_nodes)
                if m_nodes == 0:
                    continue
                mask_k = idx_cut == m_nodes
                if not np.any(mask_k):
                    continue

                k_group = k_nz[mask_k]  # (kg,)
                nz_group = nz_idx[mask_k]

                xg = x[:m_nodes]

                q = xg[None, :] / k_group[:, None]
                z = 0.5 * (q * q)
                log_r = np.log(q) - 0.5 * np.log(2.0)
                common = -z

                W = (norm * series_fac[:m_nodes] * (x_pow[:m_nodes]))[None, :] / (
                    k_group[:, None] ** denom_pow
                )

                if not is_coulomb:
                    if is_constant:
                        Veff = np.full_like(q, kappa / (2.0 * np.pi), dtype=np.float64)
                    else:
                        assert pot_fn is not None
                        Vraw = np.asarray(pot_fn(q.reshape(-1)))
                        if np.iscomplexobj(Vraw):
                            raise ValueError("Callable potential must be real-valued.")
                        Vraw = Vraw.astype(np.float64, copy=False).reshape(q.shape)
                        Veff = Vraw / (2.0 * np.pi)
                else:
                    Veff = None

                WB_ds: list[RealArray | None] = [None] * (max_d_sum + 1)
                for ds in ds_needed:
                    ds = int(ds)
                    power = (ds - 1) if is_coulomb else ds
                    base_real = np.exp(common + float(power) * log_r)
                    if not is_coulomb:
                        base_real = base_real * Veff
                    WB_ds[ds] = base_real * W

                for N in Ns_here:
                    quad_list = buckets.get(int(N), [])
                    if not quad_list:
                        continue

                    N_idx = int(N - minN)
                    signN = int(parity[N_idx])
                    phase_N = phase_table[N_idx, nz_group]

                    nQ = len(quad_list)
                    n1_arr = np.fromiter((q[0] for q in quad_list), dtype=int, count=nQ)
                    m1_arr = np.fromiter((q[1] for q in quad_list), dtype=int, count=nQ)
                    n2_arr = np.fromiter((q[2] for q in quad_list), dtype=int, count=nQ)
                    m2_arr = np.fromiter((q[3] for q in quad_list), dtype=int, count=nQ)
                    sel_idx_arr = np.fromiter((q[4] for q in quad_list), dtype=int, count=nQ)

                    d1 = d_nm[n1_arr, m1_arr].astype(int)
                    d2 = d_nm[m2_arr, n2_arr].astype(int)
                    ds = (d1 + d2).astype(int)

                    nrm = (nrm_d[d1] * nrm_d[d2]) / (
                        nrm_lag[n1_arr, m1_arr] * nrm_lag[n2_arr, m2_arr]
                    )
                    pref = (phase_power[ds] * nrm).astype(np.complex128)
                    if is_coulomb:
                        pref = pref * (float(kappa) / sqrt2)

                    extra_sgns = extra_sign_nm[n2_arr, m2_arr].astype(np.int8)
                    scalar = (pref * extra_sgns.astype(np.float64) * float(signN)).astype(
                        np.complex128
                    )

                    key1_g = key_nm[n1_arr, m1_arr].astype(int)
                    key2_g = key_nm[m2_arr, n2_arr].astype(int)

                    for start in range(0, nQ, int(chunk_size)):
                        end = min(start + int(chunk_size), nQ)
                        sl = slice(start, end)
                        b = end - start

                        ds_b = ds[sl]
                        scalar_b = scalar[sl]
                        sel_idx_b = sel_idx_arr[sl]

                        radial = np.empty((k_group.size, b), dtype=np.float64)
                        radial.fill(0.0)

                        for ds_val in np.unique(ds_b):
                            ds_val = int(ds_val)
                            WB = WB_ds[ds_val]
                            if WB is None:
                                power = (ds_val - 1) if is_coulomb else ds_val
                                base_real = np.exp(common + float(power) * log_r)
                                if not is_coulomb:
                                    base_real = base_real * Veff
                                WB = base_real * W
                                WB_ds[ds_val] = WB

                            mask = ds_b == ds_val
                            if not np.any(mask):
                                continue
                            mask_idx = np.nonzero(mask)[0]

                            # Small per-chunk cache for Laguerre values on z (=q^2/2)
                            L_cache: OrderedDict[int, RealArray] = OrderedDict()
                            cache_max = 32

                            def _get_L(
                                key: int,
                                _L_cache: OrderedDict[int, RealArray] = L_cache,
                                _z: RealArray = z,
                                _cache_max: int = cache_max,
                            ) -> RealArray:
                                if key in _L_cache:
                                    _L_cache.move_to_end(key)
                                    return _L_cache[key]
                                p = int(key_p[key])
                                d = int(key_d[key])
                                L_val = cast(RealArray, sps.eval_genlaguerre(p, d, _z))
                                _L_cache[key] = L_val
                                if len(_L_cache) > _cache_max:
                                    _L_cache.popitem(last=False)
                                return L_val

                            key1_g_b = key1_g[sl]
                            key2_g_b = key2_g[sl]

                            for j_rel, idx_rel in enumerate(mask_idx):
                                k1 = int(key1_g_b[mask][j_rel])
                                k2 = int(key2_g_b[mask][j_rel])
                                L1 = _get_L(k1)
                                L2 = _get_L(k2)
                                radial[:, idx_rel] = np.sum(WB * L1 * L2, axis=1)

                        val = (phase_N[:, None] * radial) * scalar_b[None, :]

                        Xs[np.ix_(nz_group, sel_idx_b)] = val

    # Fill k=0 via generalized Gauss-Laguerre (only N=0 contributes)
    if z_idx.size > 0:
        quad_list = buckets.get(0, [])
        if quad_list:
            X0 = np.zeros((len(select_list),), dtype=np.complex128)
            n1_0 = np.fromiter((q[0] for q in quad_list), dtype=int, count=len(quad_list))
            m1_0 = np.fromiter((q[1] for q in quad_list), dtype=int, count=len(quad_list))
            n2_0 = np.fromiter((q[2] for q in quad_list), dtype=int, count=len(quad_list))
            m2_0 = np.fromiter((q[3] for q in quad_list), dtype=int, count=len(quad_list))
            sel_idx_0 = np.fromiter((q[4] for q in quad_list), dtype=int, count=len(quad_list))

            d1 = d_nm[n1_0, m1_0].astype(int)
            d2 = d_nm[m2_0, n2_0].astype(int)
            ds = (d1 + d2).astype(int)

            nrm = (nrm_d[d1] * nrm_d[d2]) / (nrm_lag[n1_0, m1_0] * nrm_lag[n2_0, m2_0])
            pref = (phase_power[ds] * nrm).astype(np.complex128)
            if is_coulomb:
                pref = pref * (float(kappa) / sqrt2)

            extra_sgns = extra_sign_nm[n2_0, m2_0].astype(np.int8)
            scalar = (pref * extra_sgns.astype(np.float64)).astype(np.complex128)

            key1 = key_nm[n1_0, m1_0].astype(int)
            key2 = key_nm[m2_0, n2_0].astype(int)

            for ds_val in np.unique(ds):
                ds_val = int(ds_val)
                alpha = 0.5 * (ds_val - 1) if is_coulomb else 0.5 * ds_val
                xg, wg = sps.roots_genlaguerre(int(hankel_nlag), alpha)
                if not is_coulomb:
                    if is_constant:
                        w_eff = wg * (kappa / (2.0 * np.pi))
                    else:
                        assert pot_fn is not None
                        Vraw = pot_fn(np.sqrt(2.0 * xg))
                        Vraw = np.asarray(Vraw)
                        w_eff = wg * (Vraw / (2.0 * np.pi))
                else:
                    w_eff = wg

                mask = ds == ds_val
                if not np.any(mask):
                    continue

                keys = np.unique(np.concatenate([key1[mask], key2[mask]]).astype(int))
                L_vals = np.empty((xg.size, keys.size), dtype=np.float64)
                for j, key in enumerate(keys):
                    p = int(key_p[key])
                    d = int(key_d[key])
                    L_vals[:, j] = sps.eval_genlaguerre(p, d, xg)
                key_map = {int(k): j for j, k in enumerate(keys)}
                L1 = L_vals[:, [key_map[int(k)] for k in key1[mask]]]
                L2 = L_vals[:, [key_map[int(k)] for k in key2[mask]]]
                radial = np.sum(w_eff[:, None] * L1 * L2, axis=0)

                X0[sel_idx_0[mask]] = radial * scalar[mask]

            Xs[z_idx] = X0

    if sign_magneticfield == 1:
        phase1 = 1 - 2 * ((sel_n1 - sel_m1) & 1)
        phase2 = 1 - 2 * ((sel_n2 - sel_m2) & 1)
        Xs = cast(ComplexArray, Xs.conj() * (phase1 * phase2)[None, :])

    return Xs, select_list


__all__ = ["get_exchange_kernels_hankel"]
