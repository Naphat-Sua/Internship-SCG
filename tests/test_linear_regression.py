import numpy as np

from conftest import load_module

lr = load_module("Linear-Regression/Regression.py")


def make_line_data(w0=1.0, w1=2.0, n=50):
    X = np.linspace(-1, 1, n).reshape(-1, 1)
    Y = w0 + w1 * X
    return X, Y


def test_add_one_prepends_bias_column():
    X = np.array([[5.0], [7.0]])
    X1 = lr.add_one(X)
    assert X1.shape == (2, 2)
    assert np.array_equal(X1[:, 0], [1.0, 1.0])
    assert np.array_equal(X1[:, 1], [5.0, 7.0])


def test_normal_equation_recovers_exact_coefficients(capsys):
    X, Y = make_line_data(w0=1.0, w1=2.0)
    predict = lr.predict_example1(X, Y)
    capsys.readouterr()
    assert np.allclose(predict, Y, atol=1e-8)


def test_sklearn_example_recovers_exact_coefficients(capsys):
    X, Y = make_line_data(w0=-3.0, w1=0.5)
    predict = lr.predict_example2(X, Y)
    capsys.readouterr()
    assert np.allclose(predict.reshape(-1, 1), Y, atol=1e-8)


def test_gradient_descent_converges_close_to_solution(capsys):
    # standardized data keeps the hand-written gradient descent stable
    X, Y = make_line_data(w0=0.0, w1=2.0)
    FX_List, Acc_List, Loss_List = lr.predict_example5(X, Y)
    capsys.readouterr()
    assert Loss_List[-1] < 0.01          # near-zero mean squared error
    assert Acc_List[-1] > 99.0           # near-perfect R2 score
    assert Loss_List[-1] <= Loss_List[0]  # loss must not get worse


def test_polyfit_and_linregress_agree(capsys):
    X, Y = make_line_data(w0=4.0, w1=-1.5)
    p6 = lr.predict_example6(X.copy(), Y.copy())
    p7 = lr.predict_example7(X.copy(), Y.copy())
    capsys.readouterr()
    assert np.allclose(p6, p7, atol=1e-8)
    assert np.allclose(p6.reshape(-1, 1), Y, atol=1e-8)


def test_isnan_helper():
    assert lr.isNan(np.array([1.0, np.nan]))
    assert not lr.isNan(np.array([1.0, 2.0]))
