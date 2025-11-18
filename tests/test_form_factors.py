import numpy as np

from quantumhall_matrixelements import get_form_factors


def test_form_factor_known_values():
    """Test F_00(q), F_01(q), and F_10(q) against analytic formulas."""
    q_mags = np.array([0.1, 1.0, 2.0])
    q_angles = np.array([0.0, np.pi / 4, np.pi / 2])
    nmax = 2
    lB = 1.0

    F = get_form_factors(q_mags, q_angles, nmax, lB)

    # F_00(q) = exp(-(|q|lB)²/4)
    expected_F00 = np.exp(-((q_mags * lB) ** 2) / 4.0)
    assert np.allclose(F[:, 0, 0], expected_F00)

    # Common magnitude and Gaussian terms
    mag_term = q_mags * lB / np.sqrt(2)
    gauss_term = np.exp(-((q_mags * lB) ** 2) / 4.0)

    # F_01(q) = i * (q*lB/√2) * exp(-|q|²lB²/4) * exp(iθ)
    phase_term_01 = np.exp(1j * q_angles)
    expected_F01 = 1j * mag_term * gauss_term * phase_term_01
    assert np.allclose(F[:, 0, 1], expected_F01)

    # F_10(q) = i * (q*lB/√2) * exp(-|q|²lB²/4) * exp(-iθ)
    phase_term_10 = np.exp(-1j * q_angles)
    expected_F10 = 1j * mag_term * gauss_term * phase_term_10
    assert np.allclose(F[:, 1, 0], expected_F10)


def test_form_factors_conjugation_pairing():
    nmax = 3
    # Choose two G vectors: (q,theta) and (-q,theta+pi)
    q = 0.8
    G_mags = np.array([q, q])
    theta = 0.37
    thetas = np.array([theta, theta + np.pi])
    F = get_form_factors(G_mags, thetas, nmax)
    # F(-G) should equal F(G)^† (transpose+conj) due to reversal of angle
    assert np.allclose(F[1], F[0].conj().T, atol=1e-12)


def test_form_factors_small_q_expansion_lowest_LL():
    nmax = 1
    qs = np.linspace(0.0, 1e-3, 4)
    thetas = np.zeros_like(qs)
    F = get_form_factors(qs, thetas, nmax)[:, 0, 0]
    # For n=0: F(q)=exp(-q^2/4) ≈ 1 - q^2/4
    approx = 1 - (qs**2) / 4
    assert np.allclose(F, approx, rtol=0, atol=1e-10)


def test_form_factors_offdiag_scaling_power():
    nmax = 3
    qs = np.array([1e-4])
    thetas = np.array([0.0])
    F = get_form_factors(qs, thetas, nmax)[0]
    # Off-diagonal magnitudes scale ~ q^{|Δn|}
    q = qs[0]
    for n in range(nmax):
        for m in range(nmax):
            diff = abs(n - m)
            if diff == 0:
                continue
            if abs(F[n, m]) > 0:
                ratio = abs(F[n, m]) / (q**diff)
                assert ratio < 10  # Loose upper bound; prevents wrong power
                assert ratio > 1e-6  # Not numerically underflowed

