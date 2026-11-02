"""Аналитические тесты вероятностей, энтропии и шума (шаг 6)."""
import numpy as np
import pytest

from analyzer.services import probs as P


def test_sigmoid_values_and_symmetry():
    z = np.array([-800.0, -2.0, 0.0, 2.0, 800.0])
    s = P.sigmoid(z)
    assert s[2] == 0.5 and s[0] == pytest.approx(0) and s[-1] == pytest.approx(1)
    assert s[3] == pytest.approx(1 / (1 + np.exp(-2)))
    assert s[1] + s[3] == pytest.approx(1.0)  # σ(−z) = 1 − σ(z)
    assert np.all(np.isfinite(s))


def test_softmax_properties():
    z = np.array([2.0, 1.0, 0.1])
    p = P.softmax(z)
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx(np.exp(z) / np.exp(z).sum())
    assert P.softmax(z + 1000) == pytest.approx(p)  # сдвиг не меняет
    # τ → 0: вырождается в argmax; τ → ∞: в равномерное
    assert P.softmax(z, 1e-3) == pytest.approx([1, 0, 0], abs=1e-9)
    assert P.softmax(z, 1e6) == pytest.approx([1 / 3] * 3, abs=1e-5)


def test_softmax_of_two_is_sigmoid_of_difference():
    a, b = 1.7, -0.4
    assert P.softmax([a, b])[0] == pytest.approx(P.sigmoid(np.array([a - b]))[0])


def test_sigmoids_need_not_sum_to_one():
    """Многометочная модель: независимые σ(z_i) в сумме могут быть > 1."""
    assert P.sigmoid(np.array([3.0, 2.5, 2.0])).sum() > 2


def test_entropy_bounds():
    K = 8
    assert P.entropy(np.ones(K) / K) == pytest.approx(np.log(K))
    assert P.entropy([1.0, 0, 0]) == 0.0
    assert P.entropy([0.5, 0.5]) == pytest.approx(np.log(2))
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(K))
        assert 0 <= P.entropy(p) <= np.log(K) + 1e-12
    assert P.binary_entropy(0.5) == pytest.approx(np.log(2))
    assert P.binary_entropy(1.0) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("snr", [-5, 0, 10, 30])
def test_add_noise_exact_snr(snr):
    t = np.arange(16000) / 16000
    x = 0.3 * np.sin(2 * np.pi * 440 * t)
    y, info = P.add_noise(x, snr, seed=1)
    n = y - x
    assert 10 * np.log10(np.mean(x ** 2) / np.mean(n ** 2)) == pytest.approx(snr, abs=1e-9)
    assert info["p_noise"] == pytest.approx(np.mean(n ** 2))
    y2, _ = P.add_noise(x, snr, seed=1)
    assert np.array_equal(y, y2)  # воспроизводимо


def test_bce_gradient_is_p_minus_y():
    """∂/∂z BCE(σ(z), y) = σ(z) − y (цепное правило), проверка конечной разностью."""
    z, y, h = 0.7, 1.0, 1e-6

    def bce(z):
        p = P.sigmoid(np.array([z]))[0]
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    num = (bce(z + h) - bce(z - h)) / (2 * h)
    assert num == pytest.approx(P.sigmoid(np.array([z]))[0] - y, rel=1e-6)


def test_ce_softmax_gradient_is_p_minus_onehot():
    z = np.array([0.3, -1.2, 2.0, 0.5])
    y = np.array([0, 0, 1, 0.0])
    h = 1e-6
    ce = lambda z: -np.log(P.softmax(z)[2])  # noqa: E731
    g = np.array([(ce(z + h * e) - ce(z - h * e)) / (2 * h) for e in np.eye(4)])
    assert g == pytest.approx(P.softmax(z) - y, rel=1e-5, abs=1e-8)
