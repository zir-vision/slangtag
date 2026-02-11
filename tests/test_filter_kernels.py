import numpy as np

from tests.runtime import RUNTIME, require_support, spy


def test_filter_builds_blob_pair_extents():
    require_support("filter")

    words_per_point = 6
    words_per_extent = 11
    points = np.array(
        [
            [3, 7, 10, 20, 0, 1],
            [3, 7, 11, 21, 2, 0],
            [5, 9, 30, 40, 3, 1],
        ],
        dtype=np.uint32,
    )

    valid_points = points.shape[0]
    points_buf = RUNTIME.device.create_buffer(
        size=valid_points * words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(points.reshape(-1)),
    )
    extents_buf = RUNTIME.device.create_buffer(
        size=valid_points * words_per_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    extent_count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    RUNTIME.filter_module.build_blob_pair_extents(
        spy.grid(shape=(1,)),
        points_buf,
        extents_buf,
        extent_count_buf,
        valid_points,
    )

    extent_count = int(extent_count_buf.to_numpy()[0])
    assert extent_count == 2

    extents = extents_buf.to_numpy()[: extent_count * words_per_extent].reshape(
        (extent_count, words_per_extent)
    )

    expected = np.array(
        [
            [3, 7, 10, 11, 20, 21, 0, 2, 0xFFFFFFF5, 1, 0xFFFFFFFF],
            [5, 9, 30, 30, 40, 40, 2, 1, 10, 0xFFFFFFFF, 1],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(extents, expected)


def test_filter_selects_blob_pair_extents_and_rewrites_offsets():
    require_support("filter")

    words_per_extent = 11
    extents = np.array(
        [
            [1, 2, 10, 20, 30, 45, 0, 30, 100, 2, 1],
            [3, 4, 5, 7, 8, 9, 30, 10, 0, 0, 0],
            [5, 6, 2, 15, 4, 18, 40, 25, 50, 1, 1],
        ],
        dtype=np.uint32,
    )
    extent_count = extents.shape[0]

    extents_buf = RUNTIME.device.create_buffer(
        size=extent_count * words_per_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(extents.reshape(-1)),
    )
    filtered_extents_buf = RUNTIME.device.create_buffer(
        size=extent_count * words_per_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    selected_extent_count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )
    selected_point_count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    RUNTIME.filter_module.filter_blob_pair_extents(
        spy.grid(shape=(1,)),
        extents_buf,
        filtered_extents_buf,
        selected_extent_count_buf,
        selected_point_count_buf,
        extent_count,
        3,
        1,
        1,
        24,
        100,
    )

    selected_extent_count = int(selected_extent_count_buf.to_numpy()[0])
    selected_point_count = int(selected_point_count_buf.to_numpy()[0])
    assert selected_extent_count == 2
    assert selected_point_count == 55

    filtered_extents = filtered_extents_buf.to_numpy().reshape((extent_count, words_per_extent))
    expected = np.array(
        [
            [1, 2, 10, 20, 30, 45, 0, 30, 100, 2, 1],
            [3, 4, 5, 7, 8, 9, 30, 0, 0, 0, 0],
            [5, 6, 2, 15, 4, 18, 30, 25, 50, 1, 1],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(filtered_extents, expected)


def test_filter_rewrites_selected_blob_points_with_theta():
    require_support("filter")

    words_per_point = 6
    words_per_extent = 11
    selected_words_per_point = 4
    valid_points = 3
    extent_count = 2
    selected_point_count = 2

    sorted_points = np.array(
        [
            [1, 2, 1, 1, 0, 1],
            [3, 4, 4, 4, 0, 1],
            [3, 4, 6, 4, 0, 1],
        ],
        dtype=np.uint32,
    )
    extents = np.array(
        [
            [1, 2, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [3, 4, 4, 6, 4, 4, 1, 2, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    filtered_extents = np.array(
        [
            [1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [3, 4, 4, 6, 4, 4, 0, 2, 0, 0, 0],
        ],
        dtype=np.uint32,
    )

    sorted_points_buf = RUNTIME.device.create_buffer(
        size=valid_points * words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(sorted_points.reshape(-1)),
    )
    extents_buf = RUNTIME.device.create_buffer(
        size=extent_count * words_per_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(extents.reshape(-1)),
    )
    filtered_extents_buf = RUNTIME.device.create_buffer(
        size=extent_count * words_per_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(filtered_extents.reshape(-1)),
    )
    selected_points_buf = RUNTIME.device.create_buffer(
        size=selected_point_count * selected_words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.filter_module.rewrite_selected_blob_points_with_theta(
        spy.grid(shape=(valid_points,)),
        sorted_points_buf,
        extents_buf,
        filtered_extents_buf,
        selected_points_buf,
        extent_count,
        valid_points,
    )

    selected_points = selected_points_buf.to_numpy().reshape(
        (selected_point_count, selected_words_per_point)
    )

    # Match shader float precision and constants for exact theta quantization.
    cx = np.float32(0.5) * (np.float32(4.0) + np.float32(6.0)) + np.float32(0.05118)
    cy = np.float32(0.5) * (np.float32(4.0) + np.float32(4.0)) - np.float32(0.028581)
    theta_scale = np.float32(8.0e6)
    pi = np.float32(np.pi)
    theta0 = int(max(0.0, float((np.arctan2(np.float32(4.0) - cy, np.float32(4.0) - cx) + pi) * theta_scale)) + 0.5)
    theta1 = int(max(0.0, float((np.arctan2(np.float32(4.0) - cy, np.float32(6.0) - cx) + pi) * theta_scale)) + 0.5)
    expected = np.array(
        [
            [1, theta0, 4, 4],
            [1, theta1, 6, 4],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(selected_points[:, [0, 2, 3]], expected[:, [0, 2, 3]])
    np.testing.assert_allclose(
        selected_points[:, 1].astype(np.int64),
        expected[:, 1].astype(np.int64),
        atol=8,
        rtol=0.0,
    )


def test_filter_builds_line_fit_points_prefix():
    require_support("filter")

    words_per_selected_point = 4
    words_per_line_fit_point = 10
    selected_points = np.array(
        [
            [0, 100, 2, 2],
            [0, 200, 4, 2],
            [1, 100, 2, 4],
        ],
        dtype=np.uint32,
    )
    point_count = selected_points.shape[0]

    selected_points_buf = RUNTIME.device.create_buffer(
        size=point_count * words_per_selected_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(selected_points.reshape(-1)),
    )
    decimated_tex = RUNTIME.device.create_texture(
        width=6,
        height=6,
        format=spy.Format.r8_uint,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=np.zeros((6, 6), dtype=np.uint8),
    )
    line_fit_points_buf = RUNTIME.device.create_buffer(
        size=point_count * words_per_line_fit_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.filter_module.build_line_fit_points(
        spy.grid(shape=(1,)),
        selected_points_buf,
        decimated_tex,
        line_fit_points_buf,
        point_count,
        6,
        6,
        2,
    )

    line_fit_points = line_fit_points_buf.to_numpy().reshape((point_count, words_per_line_fit_point))

    def decode_i64(lo_word, hi_word):
        value = int(lo_word) | (int(hi_word) << 32)
        if value >= (1 << 63):
            value -= 1 << 64
        return value

    expected_scalars = np.array(
        [
            [5, 5, 1, 0],
            [14, 10, 2, 0],
            [5, 9, 1, 1],
        ],
        dtype=np.int64,
    )
    mx = np.array([np.int32(v).item() for v in line_fit_points[:, 0]], dtype=np.int64)
    my = np.array([np.int32(v).item() for v in line_fit_points[:, 1]], dtype=np.int64)
    w = np.array([np.int32(v).item() for v in line_fit_points[:, 2]], dtype=np.int64)
    blob = line_fit_points[:, 3].astype(np.int64)
    got_scalars = np.stack([mx, my, w, blob], axis=1)
    np.testing.assert_array_equal(got_scalars, expected_scalars)

    expected_mxx = np.array([25, 106, 25], dtype=np.int64)
    expected_myy = np.array([25, 50, 81], dtype=np.int64)
    expected_mxy = np.array([25, 70, 45], dtype=np.int64)
    got_mxx = np.array([decode_i64(row[4], row[5]) for row in line_fit_points], dtype=np.int64)
    got_myy = np.array([decode_i64(row[6], row[7]) for row in line_fit_points], dtype=np.int64)
    got_mxy = np.array([decode_i64(row[8], row[9]) for row in line_fit_points], dtype=np.int64)
    np.testing.assert_array_equal(got_mxx, expected_mxx)
    np.testing.assert_array_equal(got_myy, expected_myy)
    np.testing.assert_array_equal(got_mxy, expected_mxy)


def test_filter_fits_line_errors_and_outputs_peak_records():
    require_support("filter")

    words_per_line_fit_point = 10
    words_per_extent = 11
    words_per_peak = 3
    point_count = 3

    line_fit_points = np.array(
        [
            [np.uint32(np.int32(3)), np.uint32(np.int32(3)), np.uint32(np.int32(1)), 0, 9, 0, 9, 0, 9, 0],
            [np.uint32(np.int32(8)), np.uint32(np.int32(6)), np.uint32(np.int32(2)), 0, 34, 0, 18, 0, 24, 0],
            [np.uint32(np.int32(15)), np.uint32(np.int32(9)), np.uint32(np.int32(3)), 0, 83, 0, 27, 0, 45, 0],
        ],
        dtype=np.uint32,
    )
    extents = np.array(
        [[1, 2, 0, 10, 0, 10, 0, 3, 0, 0, 0]],
        dtype=np.uint32,
    )

    line_fit_points_buf = RUNTIME.device.create_buffer(
        size=point_count * words_per_line_fit_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(line_fit_points.reshape(-1)),
    )
    extents_buf = RUNTIME.device.create_buffer(
        size=words_per_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(extents.reshape(-1)),
    )
    errs_buf = RUNTIME.device.create_buffer(
        size=point_count * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    filtered_errs_buf = RUNTIME.device.create_buffer(
        size=point_count * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    peaks_buf = RUNTIME.device.create_buffer(
        size=point_count * words_per_peak * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.filter_module.fit_line_errors_and_peaks(
        spy.grid(shape=(1,)),
        line_fit_points_buf,
        extents_buf,
        errs_buf,
        filtered_errs_buf,
        peaks_buf,
        1,
        point_count,
    )

    errs = errs_buf.to_numpy().view(np.float32)
    filtered_errs = filtered_errs_buf.to_numpy().view(np.float32)
    peaks = peaks_buf.to_numpy().reshape((point_count, words_per_peak))

    assert np.all(np.isfinite(errs))
    assert np.all(np.isfinite(filtered_errs))
    np.testing.assert_array_equal(peaks[:, 2], np.array([0, 1, 2], dtype=np.uint32))
    assert np.all((peaks[:, 0] == 0xFFFF) | (peaks[:, 0] == 0))


def test_filter_builds_peak_extents():
    require_support("filter")

    words_per_peak = 3
    words_per_peak_extent = 3
    peak_errors = np.array([-0.9, -0.6, -0.4, -0.2, -0.1], dtype=np.float32).view(np.uint32)
    sorted_peaks = np.array(
        [
            [1, peak_errors[0], 11],
            [1, peak_errors[1], 14],
            [3, peak_errors[2], 21],
            [3, peak_errors[3], 28],
            [9, peak_errors[4], 37],
        ],
        dtype=np.uint32,
    )
    valid_peaks = sorted_peaks.shape[0]

    sorted_peaks_buf = RUNTIME.device.create_buffer(
        size=valid_peaks * words_per_peak * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(sorted_peaks.reshape(-1)),
    )
    peak_extents_buf = RUNTIME.device.create_buffer(
        size=valid_peaks * words_per_peak_extent * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    peak_extent_count_buf = RUNTIME.device.create_buffer(
        size=4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    RUNTIME.filter_module.build_peak_extents(
        spy.grid(shape=(1,)),
        sorted_peaks_buf,
        peak_extents_buf,
        peak_extent_count_buf,
        valid_peaks,
    )

    peak_extent_count = int(peak_extent_count_buf.to_numpy()[0])
    assert peak_extent_count == 3
    peak_extents = peak_extents_buf.to_numpy()[: peak_extent_count * words_per_peak_extent].reshape(
        (peak_extent_count, words_per_peak_extent)
    )
    expected = np.array(
        [
            [1, 0, 2],
            [3, 2, 2],
            [9, 4, 1],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(peak_extents, expected)
