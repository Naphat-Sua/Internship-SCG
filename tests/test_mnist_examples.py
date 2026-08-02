import numpy as np

from conftest import load_module

mn = load_module("MNIST/Dataset.py")


def test_encode_shapes_and_one_hot():
    Y = np.array([0, 3, 9])
    encoded = mn.encode(Y)
    assert encoded.shape == (3, 10)
    assert np.array_equal(encoded[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(encoded[1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert encoded.sum() == 3  # exactly one hot bit per row


def test_encode_decode_roundtrip():
    Y = np.arange(10)
    assert np.array_equal(mn.decode(mn.encode(Y)), Y)


def test_restore_img_shape():
    X = np.arange(2 * 64).reshape(2, 64)
    images = mn.restoreImg(X)
    assert images.shape == (2, 8, 8)
    assert np.array_equal(images[0][0], np.arange(8))


def test_nearest_neighbors_is_accurate_on_digits():
    Xtrain, Xtest, Ytrain, Ytest = mn.getDatasets()
    accuracy = mn.train_nearest_neighbors(Xtrain, Ytrain, Xtest, Ytest)
    # vectorized 1-NN with Manhattan distance is strong on the digits dataset
    assert accuracy > 95.0


def test_support_vector_is_accurate_on_digits():
    Xtrain, Xtest, Ytrain, Ytest = mn.getDatasets()
    accuracy = mn.train_support_vector(Xtrain, Ytrain, Xtest, Ytest)
    assert accuracy > 95.0
