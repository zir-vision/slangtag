from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

K_EXTENT_MIN_X_WORD = 2
K_EXTENT_MAX_X_WORD = 3
K_EXTENT_MIN_Y_WORD = 4
K_EXTENT_MAX_Y_WORD = 5
K_EXTENT_STARTING_OFFSET_WORD = 6
K_EXTENT_COUNT_WORD = 7
K_EXTENT_PXGX_PYGY_SUM_WORD = 8
K_EXTENT_GX_SUM_WORD = 9
K_EXTENT_GY_SUM_WORD = 10

K_PEAK_BLOB_INDEX_WORD = 0
K_PEAK_POINT_INDEX_WORD = 2

K_PEAK_EXTENT_BLOB_INDEX_WORD = 0
K_PEAK_EXTENT_STARTING_OFFSET_WORD = 1
K_PEAK_EXTENT_COUNT_WORD = 2

K_FITTED_QUAD_WORDS_PER_QUAD = 15
K_MAX_QUAD_LOCAL_PEAK_INDICES = 32


@dataclass
class QuadLineFitMoments:
    N: int = 0
    Mx: int = 0
    My: int = 0
    W: int = 0
    Mxx: int = 0
    Myy: int = 0
    Mxy: int = 0


@dataclass
class FitQuadCandidate:
    valid: bool = False
    blob_index: int = 0
    score: float = 0.0
    indices: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    moments: list[QuadLineFitMoments] = field(
        default_factory=lambda: [QuadLineFitMoments() for _ in range(4)]
    )


def _u32_to_i32(word: int) -> int:
    value = int(word) & 0xFFFFFFFF
    if value & 0x80000000:
        value -= 0x100000000
    return value


def _u32_pair_to_i64(lo: int, hi: int) -> int:
    bits = ((int(hi) & 0xFFFFFFFF) << 32) | (int(lo) & 0xFFFFFFFF)
    if bits & (1 << 63):
        bits -= 1 << 64
    return bits


def _f32_to_u32(value: float) -> np.uint32:
    return np.asarray([value], dtype=np.float32).view(np.uint32)[0]


def _blob_extent_dot(filtered_blob_extents: np.ndarray, blob_index: int) -> float:
    extent = filtered_blob_extents[blob_index]
    pxgx_plus_pygy_sum = _u32_to_i32(extent[K_EXTENT_PXGX_PYGY_SUM_WORD])
    gx_sum = _u32_to_i32(extent[K_EXTENT_GX_SUM_WORD])
    gy_sum = _u32_to_i32(extent[K_EXTENT_GY_SUM_WORD])
    min_x = int(extent[K_EXTENT_MIN_X_WORD])
    max_x = int(extent[K_EXTENT_MAX_X_WORD])
    min_y = int(extent[K_EXTENT_MIN_Y_WORD])
    max_y = int(extent[K_EXTENT_MAX_Y_WORD])

    return (
        float(
            pxgx_plus_pygy_sum * 2
            - (min_x + max_x) * gx_sum
            - (min_y + max_y) * gy_sum
        )
        * 0.5
        - 0.05118 * float(gx_sum)
        + 0.028581 * float(gy_sum)
    )


def build_peak_extents_cpu(
    sorted_peaks: np.ndarray, valid_peaks: int | None = None
) -> np.ndarray:
    peaks = np.ascontiguousarray(sorted_peaks, dtype=np.uint32).reshape((-1, 3))
    if valid_peaks is None:
        valid_peaks = peaks.shape[0]
    valid_peaks = max(0, min(int(valid_peaks), peaks.shape[0]))
    if valid_peaks == 0:
        return np.zeros((0, 3), dtype=np.uint32)

    extents: list[tuple[int, int, int]] = []
    i = 0
    while i < valid_peaks:
        blob_index = int(peaks[i, K_PEAK_BLOB_INDEX_WORD])
        start = i
        count = 1
        while i + count < valid_peaks:
            if int(peaks[i + count, K_PEAK_BLOB_INDEX_WORD]) != blob_index:
                break
            count += 1
        extents.append((blob_index, start, count))
        i += count

    if not extents:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.asarray(extents, dtype=np.uint32)


def _decode_line_fit_prefix(line_fit_points: np.ndarray) -> tuple[np.ndarray, ...]:
    points = np.ascontiguousarray(line_fit_points, dtype=np.uint32).reshape((-1, 10))

    mx = points[:, 0].view(np.int32).astype(np.int64)
    my = points[:, 1].view(np.int32).astype(np.int64)
    w = points[:, 2].view(np.int32).astype(np.int64)

    mxx = (
        (points[:, 5].astype(np.uint64) << 32) | points[:, 4].astype(np.uint64)
    ).view(np.int64)
    myy = (
        (points[:, 7].astype(np.uint64) << 32) | points[:, 6].astype(np.uint64)
    ).view(np.int64)
    mxy = (
        (points[:, 9].astype(np.uint64) << 32) | points[:, 8].astype(np.uint64)
    ).view(np.int64)

    return mx, my, w, mxx, myy, mxy


def _read_prefix(
    prefix: tuple[np.ndarray, ...], index: int
) -> tuple[int, int, int, int, int, int]:
    mx, my, w, mxx, myy, mxy = prefix
    return (
        int(mx[index]),
        int(my[index]),
        int(w[index]),
        int(mxx[index]),
        int(myy[index]),
        int(mxy[index]),
    )


def _range_non_wrap_moments(
    prefix: tuple[np.ndarray, ...], blob_start: int, local_start: int, local_end: int
) -> QuadLineFitMoments:
    end_values = _read_prefix(prefix, blob_start + local_end)
    if local_start == 0:
        return QuadLineFitMoments(
            N=local_end + 1,
            Mx=end_values[0],
            My=end_values[1],
            W=end_values[2],
            Mxx=end_values[3],
            Myy=end_values[4],
            Mxy=end_values[5],
        )

    start_values = _read_prefix(prefix, blob_start + local_start - 1)
    return QuadLineFitMoments(
        N=local_end - local_start + 1,
        Mx=end_values[0] - start_values[0],
        My=end_values[1] - start_values[1],
        W=end_values[2] - start_values[2],
        Mxx=end_values[3] - start_values[3],
        Myy=end_values[4] - start_values[4],
        Mxy=end_values[5] - start_values[5],
    )


def _segment_moments(
    prefix: tuple[np.ndarray, ...],
    blob_start: int,
    blob_count: int,
    local_start: int,
    local_end: int,
) -> QuadLineFitMoments:
    if local_start <= local_end:
        return _range_non_wrap_moments(prefix, blob_start, local_start, local_end)

    sums_a = _range_non_wrap_moments(prefix, blob_start, local_start, blob_count - 1)
    sums_b = _range_non_wrap_moments(prefix, blob_start, 0, local_end)
    return QuadLineFitMoments(
        N=(blob_count - local_start) + (local_end + 1),
        Mx=sums_a.Mx + sums_b.Mx,
        My=sums_a.My + sums_b.My,
        W=sums_a.W + sums_b.W,
        Mxx=sums_a.Mxx + sums_b.Mxx,
        Myy=sums_a.Myy + sums_b.Myy,
        Mxy=sums_a.Mxy + sums_b.Mxy,
    )


def _line_fit_mse(moments: QuadLineFitMoments) -> float:
    if moments.W <= 0:
        return 0.0

    cxx = float(moments.Mxx * moments.W - moments.Mx * moments.Mx)
    cxy = float(moments.Mxy * moments.W - moments.Mx * moments.My)
    cyy = float(moments.Myy * moments.W - moments.My * moments.My)
    hypot_cached = math.sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy)
    return ((cxx + cyy) - hypot_cached) / float(moments.W * moments.W * 8.0)


def fit_quad_candidates_cpu(
    sorted_peaks: np.ndarray,
    peak_extents: np.ndarray,
    line_fit_points: np.ndarray,
    filtered_blob_extents: np.ndarray,
    max_nmaxima: int,
    max_line_fit_mse: float,
    filtered_blob_extent_count: int | None = None,
) -> list[FitQuadCandidate]:
    peaks = np.ascontiguousarray(sorted_peaks, dtype=np.uint32).reshape((-1, 3))
    extents = np.ascontiguousarray(peak_extents, dtype=np.uint32).reshape((-1, 3))
    blob_extents = np.ascontiguousarray(filtered_blob_extents, dtype=np.uint32).reshape(
        (-1, 11)
    )

    if filtered_blob_extent_count is None:
        filtered_blob_extent_count = blob_extents.shape[0]
    filtered_blob_extent_count = max(
        0, min(int(filtered_blob_extent_count), blob_extents.shape[0])
    )

    prefix = _decode_line_fit_prefix(line_fit_points)
    candidates: list[FitQuadCandidate] = []

    for peak_extent in extents:
        blob_index = int(peak_extent[K_PEAK_EXTENT_BLOB_INDEX_WORD])
        peak_start = int(peak_extent[K_PEAK_EXTENT_STARTING_OFFSET_WORD])
        peak_count = int(peak_extent[K_PEAK_EXTENT_COUNT_WORD])

        candidate = FitQuadCandidate(valid=False, blob_index=blob_index)

        if blob_index >= filtered_blob_extent_count or peak_count < 4:
            candidates.append(candidate)
            continue

        blob_start = int(blob_extents[blob_index, K_EXTENT_STARTING_OFFSET_WORD])
        blob_count = int(blob_extents[blob_index, K_EXTENT_COUNT_WORD])
        if blob_count < 8:
            candidates.append(candidate)
            continue

        local_peak_indices: list[int] = []
        seen: set[int] = set()
        maxima_to_use = min(max_nmaxima, peak_count, K_MAX_QUAD_LOCAL_PEAK_INDICES)
        for i in range(maxima_to_use):
            peak_index = peak_start + i
            if peak_index >= peaks.shape[0]:
                break

            point_index = int(peaks[peak_index, K_PEAK_POINT_INDEX_WORD])
            if point_index < blob_start:
                continue

            local_index = point_index - blob_start
            if local_index >= blob_count:
                continue
            if local_index in seen:
                continue

            seen.add(local_index)
            local_peak_indices.append(local_index)

        local_peak_indices.sort()
        local_peak_count = len(local_peak_indices)
        if local_peak_count < 4:
            candidates.append(candidate)
            continue

        has_best = False
        best_score = 0.0
        best_corner_indices = [0, 0, 0, 0]
        best_edge_moments = [QuadLineFitMoments() for _ in range(4)]

        for i0 in range(local_peak_count - 3):
            for i1 in range(i0 + 1, local_peak_count - 2):
                for i2 in range(i1 + 1, local_peak_count - 1):
                    for i3 in range(i2 + 1, local_peak_count):
                        corner_indices = [
                            local_peak_indices[i0],
                            local_peak_indices[i1],
                            local_peak_indices[i2],
                            local_peak_indices[i3],
                        ]
                        edge_start = corner_indices
                        edge_end = [
                            corner_indices[1],
                            corner_indices[2],
                            corner_indices[3],
                            corner_indices[0],
                        ]

                        valid = True
                        total_mse = 0.0
                        edge_moments: list[QuadLineFitMoments] = []
                        for edge_index in range(4):
                            moments = _segment_moments(
                                prefix,
                                blob_start,
                                blob_count,
                                edge_start[edge_index],
                                edge_end[edge_index],
                            )

                            if moments.N < 2:
                                valid = False
                                break

                            mse = _line_fit_mse(moments)
                            if not math.isfinite(mse) or mse > max_line_fit_mse:
                                valid = False
                                break

                            edge_moments.append(moments)
                            total_mse += mse

                        if not valid:
                            continue

                        if (not has_best) or total_mse < best_score:
                            has_best = True
                            best_score = total_mse
                            best_corner_indices = corner_indices.copy()
                            best_edge_moments = edge_moments.copy()

        if has_best:
            candidate.valid = True
            candidate.score = best_score
            candidate.indices = [blob_start + idx for idx in best_corner_indices]
            candidate.moments = best_edge_moments

        candidates.append(candidate)

    return candidates


def _host_fit_line(
    moments: QuadLineFitMoments,
) -> tuple[float, float, float, float] | None:
    if moments.W == 0:
        return None

    cxx = moments.Mxx * moments.W - moments.Mx * moments.Mx
    cxy = moments.Mxy * moments.W - moments.Mx * moments.My
    cyy = moments.Myy * moments.W - moments.My * moments.My

    hypot_cached = math.hypot(float(cxx - cyy), float(2 * cxy))
    nx1 = float(cxx - cyy) - hypot_cached
    ny1 = float(2 * cxy)
    m1 = nx1 * nx1 + ny1 * ny1
    nx2 = float(2 * cxy)
    ny2 = float(cyy - cxx) - hypot_cached
    m2 = nx2 * nx2 + ny2 * ny2

    if m1 > m2:
        nx = nx1
        ny = ny1
    else:
        nx = nx2
        ny = ny2

    length = math.hypot(nx, ny)
    if length == 0.0:
        return None

    return (
        float(moments.Mx) / float(moments.W * 2),
        float(moments.My) / float(moments.W * 2),
        nx / length,
        ny / length,
    )


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.hypot(*(b - a)))
    bc = float(np.hypot(*(c - b)))
    ca = float(np.hypot(*(a - c)))
    p = (ab + bc + ca) * 0.5
    return math.sqrt(max(0.0, p * (p - ab) * (p - bc) * (p - ca)))


def _adjust_pixel_centers(corners: np.ndarray, quad_decimate: float) -> None:
    if quad_decimate <= 1.0:
        return
    if abs(quad_decimate - 1.5) < 1.0e-4:
        corners *= quad_decimate
        return
    corners[:] = (corners - 0.5) * quad_decimate + 0.5


def update_fit_quads_cpu(
    candidates: Sequence[FitQuadCandidate],
    filtered_blob_extents: np.ndarray,
    cos_critical_rad: float,
    min_tag_width: int,
    quad_decimate: float,
) -> tuple[np.ndarray, int]:
    blob_extents = np.ascontiguousarray(filtered_blob_extents, dtype=np.uint32).reshape(
        (-1, 11)
    )
    packed_rows: list[np.ndarray] = []

    for quad in candidates:
        if not quad.valid:
            continue

        lines: list[tuple[float, float, float, float]] = []
        valid_lines = True
        for i in range(4):
            line = _host_fit_line(quad.moments[i])
            if line is None:
                valid_lines = False
                break
            lines.append(line)
        if not valid_lines:
            continue

        corners = np.zeros((4, 2), dtype=np.float64)
        bad_determinant = False
        for i in range(4):
            i_next = (i + 1) & 3
            a00 = lines[i][3]
            a01 = -lines[i_next][3]
            a10 = -lines[i][2]
            a11 = lines[i_next][2]
            b0 = -lines[i][0] + lines[i_next][0]
            b1 = -lines[i][1] + lines[i_next][1]

            det = a00 * a11 - a10 * a01
            if abs(det) < 0.001:
                bad_determinant = True
                break

            w00 = a11 / det
            w01 = -a01 / det
            l0 = w00 * b0 + w01 * b1
            corners[i, 0] = lines[i][0] + l0 * a00
            corners[i, 1] = lines[i][1] + l0 * a10
        if bad_determinant:
            continue

        area = _triangle_area(corners[0], corners[1], corners[2]) + _triangle_area(
            corners[2], corners[3], corners[0]
        )
        min_area = 0.95 * float(min_tag_width * min_tag_width)
        if area < min_area:
            continue

        reject_corner = False
        for i in range(4):
            i0 = i
            i1 = (i + 1) & 3
            i2 = (i + 2) & 3

            dx1 = corners[i1, 0] - corners[i0, 0]
            dy1 = corners[i1, 1] - corners[i0, 1]
            dx2 = corners[i2, 0] - corners[i1, 0]
            dy2 = corners[i2, 1] - corners[i1, 1]
            denom = math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))
            if denom == 0.0:
                reject_corner = True
                break
            cos_dtheta = (dx1 * dx2 + dy1 * dy2) / denom
            if abs(cos_dtheta) > cos_critical_rad or dx1 * dy2 < dy1 * dx2:
                reject_corner = True
                break
        if reject_corner:
            continue

        _adjust_pixel_centers(corners, quad_decimate)

        row = np.zeros((K_FITTED_QUAD_WORDS_PER_QUAD,), dtype=np.uint32)
        row[0] = np.uint32(quad.blob_index)
        row[1] = np.uint32(
            1 if _blob_extent_dot(blob_extents, quad.blob_index) < 0.0 else 0
        )
        row[2] = _f32_to_u32(quad.score)
        row[3:11] = corners.astype(np.float32).reshape((8,)).view(np.uint32)
        row[11:15] = np.asarray(quad.indices, dtype=np.uint32)
        packed_rows.append(row)

    if not packed_rows:
        return np.zeros((0,), dtype=np.uint32), 0

    packed = np.vstack(packed_rows).astype(np.uint32)
    return packed.reshape((-1,)), int(packed.shape[0])
