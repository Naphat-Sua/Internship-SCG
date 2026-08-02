import numpy as np
import pandas as pd

from conftest import load_module

logreg = load_module("Log-Regression/Function.py")


def test_sigmoid_midpoint_and_symmetry():
    assert logreg.sigmoid(0) == 0.5
    x = np.linspace(-4, 4, 33)
    assert np.allclose(logreg.sigmoid(x) + logreg.sigmoid(-x), 1.0)


def test_cost_function_penalizes_wrong_confident_predictions():
    # confident and right: almost no cost
    assert logreg.cost_function(0.999, 1) < 0.01
    assert logreg.cost_function(0.001, 0) < 0.01
    # confident and wrong: large cost
    assert logreg.cost_function(0.001, 1) > 2.0
    assert logreg.cost_function(0.999, 0) > 2.0


def test_load_dataset_normalizes_columns():
    df = logreg.load_dataset()
    assert list(df.columns) == ['X1', 'X2', 'Y']
    assert len(df) > 0
    assert set(df['Y'].unique()) == {0, 1}


def test_seperate_class_splits_labels():
    df = logreg.load_dataset()
    classA, classB = logreg.seperateClass(df)
    assert (classA['Y'] == 1).all()
    assert (classB['Y'] == 0).all()
    assert len(classA) + len(classB) == len(df)


def test_train_method_learns_a_separating_boundary(capsys):
    # small, clearly separable dataset: class = 1 when x1 + x2 > 1
    rng = np.random.RandomState(42)
    n = 40
    X1 = rng.uniform(0, 1, n)
    X2 = rng.uniform(0, 1, n)
    Y = ((X1 + X2) > 1.0).astype(int)
    data_X = pd.DataFrame({'X1': X1, 'X2': X2})

    C_List = logreg.train_method(data_X, Y.reshape(-1, 1), max_steps=20000)
    capsys.readouterr()

    w0, w1, w2 = C_List[-1]
    X = np.column_stack([np.ones(n), X1, X2])
    predictions = (logreg.sigmoid(X @ np.array([[w0], [w1], [w2]])) > 0.5).astype(int)
    accuracy = np.mean(predictions.reshape(-1) == Y)
    assert accuracy >= 0.9


def test_get_decision_func_shape():
    X1 = np.linspace(0, 1, 10)
    C_List = np.array([[0.5, 1.0, -1.0], [0.2, 2.0, -1.0]])
    FX_List = logreg.getDecisionFunc(X1, C_List)
    assert FX_List.shape == (2, 10)
    # w0 + w1*x + w2*fx == 0 on the boundary
    w0, w1, w2 = C_List[0]
    assert np.allclose(w0 + w1 * X1 + w2 * FX_List[0], 0.0)
