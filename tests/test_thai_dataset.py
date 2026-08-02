from conftest import load_module

thai = load_module("Text-Classification/thai_dataset.py")


def test_word_index_loads_from_frequency_csv():
    dict_word = thai.get_word_index()
    assert len(dict_word) > 4000
    # the two most frequent Thai words from Frequency.csv
    assert dict_word['ที่'] == 1
    assert dict_word['การ'] == 2


def test_get_index_maps_known_words_and_special_tags():
    indexes = thai.get_index(['ที่', 'การ'])
    assert indexes == [1, 2]

    # special markup gets fixed indexes
    assert thai.get_index(['<NE>กรุงเทพ</NE>']) == [5001]
    assert thai.get_index(['<AB>กทม</AB>']) == [5002]
    assert thai.get_index(['<POEM>บทกวี</POEM>']) == [5003]

    # unknown / rare words map to 0
    assert thai.get_index(['zzz-unknown-word']) == [0]


def test_create_classify_dataset_splits_80_20():
    dataset_list = [
        [[1], [2], [3], [4], [5]],   # class 0: five documents
        [[6], [7], [8], [9], [10]],  # class 1: five documents
    ]
    X_train, X_test, Y_train, Y_test = thai.create_classify_dataset(dataset_list)
    assert len(X_train) == 8   # 80% of each class
    assert len(X_test) == 2
    assert Y_train.count(0) == 4 and Y_train.count(1) == 4
    assert Y_test == [0, 1]
