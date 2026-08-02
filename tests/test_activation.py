import numpy as np

from conftest import load_module

act = load_module("Activation/Activation.py")


def test_sigmoid_known_values():
    assert act.sigmoid(0) == 0.5
    assert np.isclose(act.sigmoid(100), 1.0)
    assert np.isclose(act.sigmoid(-100), 0.0, atol=1e-12)


def test_sigmoid_matches_analytic_formula():
    x = np.linspace(-5, 5, 101)
    assert np.allclose(act.sigmoid(x), 1 / (1 + np.exp(-x)))


def test_tanh_matches_numpy():
    x = np.linspace(-5, 5, 101)
    assert np.allclose(act.tanh(x), np.tanh(x))


def test_relu():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.allclose(act.relu(x), [0.0, 0.0, 0.0, 0.5, 2.0])


def test_leaky_relu():
    x = np.array([-2.0, 0.0, 3.0])
    leak = 0.2
    expected = np.where(x > 0, x, leak * x)
    assert np.allclose(act.leaky_relu(x, leak=leak), expected)
