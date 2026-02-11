import numpy as np

from tests.runtime import RUNTIME, require_support, spy


def test_sort_orders_selected_blob_points_by_blob_and_theta():
    require_support("sort")

    words_per_point = 4
    valid_points = 4
    total_points = 4
    points = np.array(
        [
            [2, 20, 7, 7],
            [1, 30, 3, 3],
            [1, 10, 2, 2],
            [2, 5, 6, 6],
        ],
        dtype=np.uint32,
    )

    input_buf = RUNTIME.device.create_buffer(
        size=valid_points * words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(points.reshape(-1)),
    )
    sorted_buf = RUNTIME.device.create_buffer(
        size=total_points * words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.sort_module.prepare_selected_blob_points(
        spy.grid(shape=(total_points,)),
        input_buf,
        sorted_buf,
        valid_points,
        total_points,
    )

    k = 2
    while k <= total_points:
        j = k // 2
        while j > 0:
            RUNTIME.sort_module.bitonic_sort_selected_blob_points(
                spy.grid(shape=(total_points,)),
                sorted_buf,
                total_points,
                j,
                k,
            )
            j //= 2
        k *= 2

    sorted_points = sorted_buf.to_numpy().reshape((valid_points, words_per_point))
    expected = np.array(
        [
            [1, 10, 2, 2],
            [1, 30, 3, 3],
            [2, 5, 6, 6],
            [2, 20, 7, 7],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(sorted_points, expected)


def test_sort_orders_peaks_by_blob_and_error():
    require_support("sort")

    words_per_peak = 3
    valid_peaks = 4
    total_peaks = 4
    errors = np.array([-0.3, -0.1, -0.8, -0.5], dtype=np.float32).view(np.uint32)
    peaks = np.array(
        [
            [2, errors[0], 7],
            [1, errors[1], 3],
            [1, errors[2], 2],
            [2, errors[3], 6],
        ],
        dtype=np.uint32,
    )

    input_buf = RUNTIME.device.create_buffer(
        size=valid_peaks * words_per_peak * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(peaks.reshape(-1)),
    )
    sorted_buf = RUNTIME.device.create_buffer(
        size=total_peaks * words_per_peak * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.sort_module.prepare_peaks(
        spy.grid(shape=(total_peaks,)),
        input_buf,
        sorted_buf,
        valid_peaks,
        total_peaks,
    )

    k = 2
    while k <= total_peaks:
        j = k // 2
        while j > 0:
            RUNTIME.sort_module.bitonic_sort_peaks(
                spy.grid(shape=(total_peaks,)),
                sorted_buf,
                total_peaks,
                j,
                k,
            )
            j //= 2
        k *= 2

    sorted_peaks = sorted_buf.to_numpy().reshape((valid_peaks, words_per_peak))
    expected = np.array(
        [
            [1, errors[2], 2],
            [1, errors[1], 3],
            [2, errors[3], 6],
            [2, errors[0], 7],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(sorted_peaks, expected)
