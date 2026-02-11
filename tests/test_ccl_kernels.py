import numpy as np

from tests.runtime import RUNTIME, require_support, run_blob_diff, run_init, spy


def test_compression_compresses_parent_chains():
    require_support("compression")

    labels_data = np.arange(16, dtype=np.uint32)
    labels_data[0] = 4
    labels_data[4] = 8
    labels_data[8] = 8
    labels_data[5] = 13
    labels_data[13] = 13
    labels_data[6] = 10
    labels_data[10] = 10
    labels_data[14] = 2
    labels_data[2] = 2

    labels = RUNTIME.device.create_buffer(
        size=16 * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=labels_data,
    )

    RUNTIME.ccl_module.compression(spy.grid(shape=(2, 2)), labels, spy.uint2(4, 4))
    result = labels.to_numpy()

    assert int(result[0]) == 8
    assert int(result[4]) == 8
    assert int(result[5]) == 13
    assert int(result[6]) == 10
    assert int(result[14]) == 2


def test_init_all_white_block():
    require_support("init")

    image = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    result = run_init(image)

    expected = np.array([0, 0b1111, 2, 3], dtype=np.uint32)
    np.testing.assert_array_equal(result, expected)


def test_init_all_black_block():
    require_support("init")

    image = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    result = run_init(image)

    left_bg = (1 << 0) | (1 << 2)
    right_bg = (1 << 1) | (1 << 3)
    packed = (left_bg << 8) | (right_bg << 16)
    expected = np.array([0, packed, 2, 2], dtype=np.uint32)
    np.testing.assert_array_equal(result, expected)


def test_init_checkerboard_block():
    require_support("init")

    image = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    result = run_init(image)

    fg = (1 << 0) | (1 << 3)
    left_bg = 1 << 2
    right_bg = 1 << 1
    packed = fg | (left_bg << 8) | (right_bg << 16)
    expected = np.array([0, packed, 2, 3], dtype=np.uint32)
    np.testing.assert_array_equal(result, expected)


def test_blob_diff_detects_boundaries_and_deduplicates_offset3():
    require_support("blob_diff")

    image = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    blobs = np.array(
        [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [7, 7, 7, 7],
            [7, 7, 7, 7],
        ],
        dtype=np.uint32,
    )
    union_sizes = np.zeros(image.size, dtype=np.uint32)
    union_sizes[3] = 100
    union_sizes[7] = 100

    result = run_blob_diff(image, blobs, union_sizes, min_blob_size=25)

    np.testing.assert_array_equal(result[0, 0], np.zeros(6, dtype=np.uint32))
    np.testing.assert_array_equal(result[1, 0], np.array([3, 7, 1, 1, 1, 1], dtype=np.uint32))
    np.testing.assert_array_equal(result[2, 0], np.array([3, 7, 1, 1, 2, 1], dtype=np.uint32))
    np.testing.assert_array_equal(result[3, 0], np.array([3, 7, 1, 1, 3, 1], dtype=np.uint32))
    np.testing.assert_array_equal(result[3, 1], np.zeros(6, dtype=np.uint32))


def test_blob_diff_filters_small_blob_sizes():
    require_support("blob_diff")

    image = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    blobs = np.array(
        [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [7, 7, 7, 7],
            [7, 7, 7, 7],
        ],
        dtype=np.uint32,
    )
    union_sizes = np.zeros(image.size, dtype=np.uint32)
    union_sizes[3] = 10
    union_sizes[7] = 10

    result = run_blob_diff(image, blobs, union_sizes, min_blob_size=25)
    assert int(np.count_nonzero(result)) == 0
