import numpy as np

import quantumhall_matrixelements.exchange_hankel as exchange_hankel


def test_get_hankel_nodes_uses_public_quadrature_fields(monkeypatch) -> None:
    class FakeHankelTransform:
        def __init__(self, nu: int, N: int, h: float, alt: bool = False) -> None:
            assert (nu, N, h, alt) == (2, 8, 0.1, False)
            self.x = np.array([1.0, 2.0], dtype=np.float64)
            self.w = np.array([0.5, 0.25], dtype=np.float64)
            self.kernel = np.array([1.5, -0.5], dtype=np.float64)
            self.dpsi = np.array([2.0, 3.0], dtype=np.float64)

    monkeypatch.setattr(exchange_hankel, "HankelTransform", FakeHankelTransform)
    exchange_hankel._get_hankel_nodes.cache_clear()
    try:
        x, series_fac = exchange_hankel._get_hankel_nodes(2, 8, 0.1)
    finally:
        exchange_hankel._get_hankel_nodes.cache_clear()

    assert np.allclose(x, np.array([1.0, 2.0], dtype=np.float64))
    assert np.allclose(
        series_fac,
        np.pi
        * np.array([0.5, 0.25], dtype=np.float64)
        * np.array([1.5, -0.5], dtype=np.float64)
        * np.array([2.0, 3.0], dtype=np.float64),
    )
