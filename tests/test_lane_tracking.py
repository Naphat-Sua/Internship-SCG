import numpy as np

from conftest import load_module

lane = load_module("Hash/CrowndStrike.py")


def make_window(**kwargs):
    params = dict(level=0, window_shape=(20, 31), img_shape=(100, 200),
                  x_init=100, max_frozen_dur=3)
    params.update(kwargs)
    return lane.Window(**params)


def test_window_requires_odd_width():
    try:
        make_window(window_shape=(20, 30))
        assert False, "expected an exception for even width"
    except Exception:
        pass


def test_window_geometry():
    w = make_window()
    assert w.y_begin == 100 - 20  # top row: img_h - (level+1)*height
    assert w.y_end == 100
    assert w.x_begin() == 100 - 15
    assert w.x_end() == 100 + 15
    assert w.area() == 20 * 31


def test_window_mask_covers_expected_pixels():
    w = make_window()
    mask = w.get_mask()
    assert mask.shape == (100, 200)
    assert mask[85, 100] == 1  # inside the window
    assert mask[10, 100] == 0  # far above the window
    assert mask.sum() == 20 * (w.x_end() - w.x_begin())


def test_freeze_unfreeze_and_dropped():
    w = make_window(max_frozen_dur=2)
    assert w.dropped  # starts dropped (frozen_dur = max+1)
    for _ in range(3):
        w.unfreeze()
    assert not w.dropped
    w.freeze()
    w.freeze()
    w.freeze()
    assert w.frozen
    assert w.dropped  # frozen longer than max_frozen_dur


def test_window_update_finds_bright_column():
    w = make_window()
    score_img = np.zeros((100, 200))
    score_img[80:100, 140] = 10.0  # a bright lane pixel column at x=140
    w.update(score_img, x_search_range=(100, 180))
    assert w.detected
    assert w.x_measured == 140


def test_strictly_decreasing():
    assert lane.strictly_decreasing([5, 4, 3])
    assert not lane.strictly_decreasing([5, 5, 3])
    assert not lane.strictly_decreasing([3, 4, 5])


def test_argmax_between():
    arr = np.array([0, 9, 2, 7, 5])
    assert lane.argmax_between(arr, 0, 5) == 1
    assert lane.argmax_between(arr, 2, 5) == 3


def test_filter_window_list():
    windows = [make_window() for _ in range(3)]
    for w in windows:
        for _ in range(3):
            w.unfreeze()  # none dropped now
    windows[1].frozen_dur = 100  # drop the middle window

    filtered, args = lane.filter_window_list(windows, remove_dropped=True)
    assert len(filtered) == 2
    assert args == [0, 2]


def test_window_filter_tracks_measurements():
    kf = lane.WindowFilter(pos_init=50.0)
    for _ in range(10):
        kf.update(80.0)
    assert abs(kf.get_position() - 80.0) < 5.0
