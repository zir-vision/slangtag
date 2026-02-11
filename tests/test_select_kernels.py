import numpy as np

from tests.runtime import RUNTIME, require_support, spy


def test_select_counts_and_compacts_nonzero_blob_points():
    require_support("select")

    words_per_point = 6
    points = np.array(
        [
            [1, 9, 10, 11, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [5, 20, 30, 31, 2, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    total_points = points.shape[0]
    input_data = np.ascontiguousarray(points.reshape(-1))

    input_buf = RUNTIME.device.create_buffer(
        size=input_data.size * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=input_data,
    )
    count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    RUNTIME.select_module.count_nonzero_blob_diff_points(
        spy.grid(shape=(total_points,)),
        input_buf,
        count_buf,
        total_points,
    )
    count = int(count_buf.to_numpy()[0])
    assert count == 2

    output_buf = RUNTIME.device.create_buffer(
        size=count * words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    output_count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    RUNTIME.select_module.filter_nonzero_blob_diff_points(
        spy.grid(shape=(total_points,)),
        input_buf,
        output_buf,
        output_count_buf,
        total_points,
    )

    output_count = int(output_count_buf.to_numpy()[0])
    assert output_count == 2

    output = output_buf.to_numpy().reshape((output_count, words_per_point))
    expected = points[points[:, 1] != 0]
    output_sorted = output[np.lexsort(output.T[::-1])]
    expected_sorted = expected[np.lexsort(expected.T[::-1])]
    np.testing.assert_array_equal(output_sorted, expected_sorted)


def test_select_counts_and_compacts_valid_peaks():
    require_support("select")

    words_per_peak = 3
    peak_errors = np.array([-0.5, -0.2, -1.0, -0.1], dtype=np.float32).view(np.uint32)
    peaks = np.array(
        [
            [1, peak_errors[0], 4],
            [0xFFFF, peak_errors[1], 8],
            [3, peak_errors[2], 9],
            [0xFFFF, peak_errors[3], 10],
        ],
        dtype=np.uint32,
    )
    total_peaks = peaks.shape[0]

    peaks_buf = RUNTIME.device.create_buffer(
        size=total_peaks * words_per_peak * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(peaks.reshape(-1)),
    )
    count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    RUNTIME.select_module.count_valid_peaks(
        spy.grid(shape=(total_peaks,)),
        peaks_buf,
        count_buf,
        total_peaks,
    )
    count = int(count_buf.to_numpy()[0])
    assert count == 2

    compacted_buf = RUNTIME.device.create_buffer(
        size=count * words_per_peak * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    compacted_count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )
    RUNTIME.select_module.filter_valid_peaks(
        spy.grid(shape=(total_peaks,)),
        peaks_buf,
        compacted_buf,
        compacted_count_buf,
        total_peaks,
    )

    compacted_count = int(compacted_count_buf.to_numpy()[0])
    assert compacted_count == 2
    compacted = compacted_buf.to_numpy().reshape((compacted_count, words_per_peak))
    expected = np.array(
        [
            [1, peak_errors[0], 4],
            [3, peak_errors[2], 9],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(compacted, expected)
