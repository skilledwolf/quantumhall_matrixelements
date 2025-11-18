"""Debug helper to print exchange-kernel symmetry deviations.

This is not part of the public API; it exists only to make it easy to
inspect symmetry errors for different backends via

    python -m quantumhall_matrixelements._debug_symmetry
"""
from __future__ import annotations

import numpy as np

from . import get_exchange_kernels


def main() -> None:
    nmax = 2
    Gs_dimless = np.array([0.0, 1.0, 1.0])
    thetas = np.array([0.0, 0.0, np.pi])
    thetas_minus = (thetas + np.pi) % (2 * np.pi)

    methods = ["gausslag", "hankel"]

    for method in methods:
        print(f"=== method={method} ===")
        X_G = get_exchange_kernels(Gs_dimless, thetas, nmax, method=method)
        X_mG = get_exchange_kernels(Gs_dimless, thetas_minus, nmax, method=method)

        expected_mG = np.transpose(X_G, (0, 3, 4, 1, 2)).conj()
        diff_G_to_mG = float(np.max(np.abs(X_mG - expected_mG)))
        print(f"max |X(-G) - (X^T(G))†| = {diff_G_to_mG:.3e}")

        idx = np.arange(nmax)
        N = (
            idx[:, None, None, None]
            - idx[None, :, None, None]
            - idx[None, None, :, None]
            + idx[None, None, None, :]
        )
        phase = (-1.0) ** np.abs(N)
        expected_internal = phase[None, ...] * expected_mG
        diff_internal = float(np.max(np.abs(X_G - expected_internal)))
        print(
            "max |X(G) - (-1)^|N| (X^T(G))†| = "
            f"{diff_internal:.3e}",
        )


if __name__ == "__main__":  # pragma: no cover - manual debug entry point
    main()

