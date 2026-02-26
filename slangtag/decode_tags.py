from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pathlib

try:
    import slangpy as spy
except ImportError:  # pragma: no cover - exercised when slangpy is unavailable
    spy = None


def _as_gray_u8(image_gray: np.ndarray) -> np.ndarray:
    if image_gray.ndim != 2:
        raise ValueError("Expected a single-channel grayscale image.")
    if image_gray.dtype == np.uint8:
        return np.ascontiguousarray(image_gray)
    return np.ascontiguousarray(np.clip(image_gray, 0, 255).astype(np.uint8))


def _extract_bits_opencv_style(
    image_gray: np.ndarray,
    corners: np.ndarray,
    marker_size: int,
    marker_border_bits: int,
    cell_size: int,
    cell_margin_rate: float,
    min_stddev_otsu: float,
) -> np.ndarray:
    if corners.shape != (4, 2):
        raise ValueError("Expected candidate corners with shape (4, 2).")
    if marker_border_bits <= 0 or cell_size <= 0:
        raise ValueError("marker_border_bits and cell_size must be positive.")
    if not (0.0 <= cell_margin_rate <= 0.5):
        raise ValueError("cell_margin_rate must be in [0.0, 0.5].")

    marker_size_with_borders = marker_size + 2 * marker_border_bits
    result_size = marker_size_with_borders * cell_size
    cell_margin_pixels = int(cell_margin_rate * cell_size)
    cell_span = cell_size - 2 * cell_margin_pixels
    if cell_span <= 0:
        raise ValueError("cell margin leaves no pixels to evaluate per cell.")

    dst_corners = np.array(
        [
            [0.0, 0.0],
            [float(result_size - 1), 0.0],
            [float(result_size - 1), float(result_size - 1)],
            [0.0, float(result_size - 1)],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.ascontiguousarray(corners.astype(np.float32)), dst_corners
    )
    warped = cv2.warpPerspective(
        image_gray,
        transform,
        (result_size, result_size),
        flags=cv2.INTER_NEAREST,
    )

    bits = np.zeros((marker_size_with_borders, marker_size_with_borders), dtype=np.uint8)

    inner = warped[
        cell_size // 2 : warped.shape[0] - cell_size // 2,
        cell_size // 2 : warped.shape[1] - cell_size // 2,
    ]
    mean, stddev = cv2.meanStdDev(inner)
    if float(stddev[0, 0]) < float(min_stddev_otsu):
        bits.fill(1 if float(mean[0, 0]) > 127.0 else 0)
        return bits

    _, warped = cv2.threshold(warped, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    for y in range(marker_size_with_borders):
        y_start = y * cell_size + cell_margin_pixels
        for x in range(marker_size_with_borders):
            x_start = x * cell_size + cell_margin_pixels
            square = warped[y_start : y_start + cell_span, x_start : x_start + cell_span]
            white_pixels = int(cv2.countNonZero(square))
            if white_pixels > square.size // 2:
                bits[y, x] = 1

    return bits


def _get_border_errors(bits: np.ndarray, marker_size: int, border_size: int) -> int:
    size_with_borders = marker_size + 2 * border_size
    if bits.shape != (size_with_borders, size_with_borders):
        raise ValueError("bits has incompatible shape for marker_size/border_size.")

    total = 0
    for y in range(size_with_borders):
        for k in range(border_size):
            total += int(bits[y, k] != 0)
            total += int(bits[y, size_with_borders - 1 - k] != 0)
    for x in range(border_size, size_with_borders - border_size):
        for k in range(border_size):
            total += int(bits[k, x] != 0)
            total += int(bits[size_with_borders - 1 - k, x] != 0)
    return total


def decode_single_quad_opencv(
    image_gray: np.ndarray,
    corners: np.ndarray,
    dictionary: cv2.aruco.Dictionary,
    detector_params: cv2.aruco.DetectorParameters | None = None,
) -> dict[str, Any] | None:
    params = detector_params if detector_params is not None else cv2.aruco.DetectorParameters()
    image_gray = _as_gray_u8(image_gray)
    corners = np.ascontiguousarray(corners, dtype=np.float32).reshape((4, 2))

    marker_size = int(dictionary.markerSize)
    marker_border_bits = int(params.markerBorderBits)

    candidate_bits = _extract_bits_opencv_style(
        image_gray=image_gray,
        corners=corners,
        marker_size=marker_size,
        marker_border_bits=marker_border_bits,
        cell_size=int(params.perspectiveRemovePixelPerCell),
        cell_margin_rate=float(params.perspectiveRemoveIgnoredMarginPerCell),
        min_stddev_otsu=float(params.minOtsuStdDev),
    )

    max_border_errors = int(
        marker_size * marker_size * float(params.maxErroneousBitsInBorderRate)
    )
    border_errors = _get_border_errors(
        candidate_bits,
        marker_size=marker_size,
        border_size=marker_border_bits,
    )

    is_inverted = False
    if bool(params.detectInvertedMarker):
        inverted = (1 - candidate_bits).astype(np.uint8, copy=False)
        inverted_errors = _get_border_errors(
            inverted,
            marker_size=marker_size,
            border_size=marker_border_bits,
        )
        if inverted_errors < border_errors:
            candidate_bits = inverted
            border_errors = inverted_errors
            is_inverted = True

    if border_errors > max_border_errors:
        return None

    inner_bits = candidate_bits[
        marker_border_bits : marker_border_bits + marker_size,
        marker_border_bits : marker_border_bits + marker_size,
    ]
    identified, marker_id, rotation = dictionary.identify(
        inner_bits,
        float(params.errorCorrectionRate),
    )
    if not identified:
        return None

    corrected_corners = np.roll(corners.astype(np.float64), int(rotation), axis=0)
    return {
        "id": int(marker_id),
        "rotation": int(rotation),
        "corners": corrected_corners,
        "corners_raw": corners.astype(np.float64),
        "is_inverted": is_inverted,
        "border_errors": int(border_errors),
    }


def decode_tags_from_fitted_quads(
    image_gray: np.ndarray,
    fitted_quads: list[dict[str, Any]],
    dictionary: cv2.aruco.Dictionary | None = None,
    detector_params: cv2.aruco.DetectorParameters | None = None,
) -> list[dict[str, Any]]:
    dictionary = (
        dictionary
        if dictionary is not None
        else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )
    params = detector_params if detector_params is not None else cv2.aruco.DetectorParameters()

    detections: list[dict[str, Any]] = []
    for quad_index, quad in enumerate(fitted_quads):
        quad_corners = np.asarray(quad["corners"], dtype=np.float32).reshape((4, 2))
        decoded = decode_single_quad_opencv(
            image_gray=image_gray,
            corners=quad_corners,
            dictionary=dictionary,
            detector_params=params,
        )
        if decoded is None:
            continue

        decoded["quad_index"] = quad_index
        if "blob_index" in quad:
            decoded["blob_index"] = int(quad["blob_index"])
        if "reversed_border" in quad:
            decoded["reversed_border"] = bool(quad["reversed_border"])
        if "score" in quad:
            decoded["score"] = float(quad["score"])
        detections.append(decoded)

    return detections


def _load_decode_module(device):
    if spy is None:
        raise RuntimeError("slangpy is required for GPU tag decoding.")
    shader_path = pathlib.Path(__file__).resolve().parent / "shaders" / "decode.slang"
    return spy.Module.load_from_file(device, str(shader_path))


def _extract_bits_slang(
    image_gray: np.ndarray,
    corners_batch: np.ndarray,
    device,
    module,
    marker_size: int,
    marker_border_bits: int,
    cell_size: int,
    cell_margin_rate: float,
    min_stddev_otsu: float,
) -> np.ndarray:
    marker_size_with_borders = marker_size + 2 * marker_border_bits
    result_size = marker_size_with_borders * cell_size
    cell_margin_pixels = int(cell_margin_rate * cell_size)
    cell_span = cell_size - 2 * cell_margin_pixels
    if cell_span <= 0:
        raise ValueError("cell margin leaves no pixels to evaluate per cell.")

    quad_count = int(corners_batch.shape[0])
    if quad_count <= 0:
        return np.zeros((0, marker_size_with_borders, marker_size_with_borders), dtype=np.uint8)

    image_tex = device.create_texture(
        width=image_gray.shape[1],
        height=image_gray.shape[0],
        format=spy.Format.r8_uint,
        usage=spy.TextureUsage.shader_resource,
        data=np.ascontiguousarray(image_gray),
    )
    corners_words = np.ascontiguousarray(corners_batch.reshape(-1))
    corners_buf = device.create_buffer(
        size=corners_words.size * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource,
        data=np.ascontiguousarray(corners_words.view(np.uint32)),
    )
    quad_param_words_per_quad = 12
    quad_params_buf = device.create_buffer(
        size=max(1, quad_count) * quad_param_words_per_quad * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    bits_count = quad_count * marker_size_with_borders * marker_size_with_borders
    bits_buf = device.create_buffer(
        size=max(1, bits_count) * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    module.prepare_decode_quads(
        spy.grid(shape=(max(1, quad_count),)),
        image_tex,
        corners_buf,
        quad_params_buf,
        quad_count,
        marker_size_with_borders,
        cell_size,
        min_stddev_otsu,
    )
    module.extract_candidate_bits(
        spy.grid(shape=(max(1, bits_count),)),
        image_tex,
        quad_params_buf,
        bits_buf,
        quad_count,
        marker_size_with_borders,
        cell_size,
        cell_margin_pixels,
        cell_span,
    )

    bits = bits_buf.to_numpy()[:bits_count].reshape(
        (quad_count, marker_size_with_borders, marker_size_with_borders)
    )
    return bits.astype(np.uint8, copy=False)


def decode_tags_from_fitted_quads_slang(
    image_gray: np.ndarray,
    fitted_quads: list[dict[str, Any]],
    device,
    dictionary: cv2.aruco.Dictionary | None = None,
    detector_params: cv2.aruco.DetectorParameters | None = None,
    module=None,
    cpu_fallback_on_miss: bool = False,
) -> list[dict[str, Any]]:
    if spy is None:
        raise RuntimeError("slangpy is not installed; cannot use GPU tag decoder.")

    dictionary = (
        dictionary
        if dictionary is not None
        else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )
    params = detector_params if detector_params is not None else cv2.aruco.DetectorParameters()
    image_gray = _as_gray_u8(image_gray)
    marker_size = int(dictionary.markerSize)
    marker_border_bits = int(params.markerBorderBits)
    cell_size = int(params.perspectiveRemovePixelPerCell)
    cell_margin_rate = float(params.perspectiveRemoveIgnoredMarginPerCell)
    module = module if module is not None else _load_decode_module(device)

    if not fitted_quads:
        return []

    corners_batch = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(quad["corners"], dtype=np.float32).reshape((4, 2))
                for quad in fitted_quads
            ],
            axis=0,
        ),
        dtype=np.float32,
    )
    candidate_bits_batch = _extract_bits_slang(
        image_gray=image_gray,
        corners_batch=corners_batch,
        device=device,
        module=module,
        marker_size=marker_size,
        marker_border_bits=marker_border_bits,
        cell_size=cell_size,
        cell_margin_rate=cell_margin_rate,
        min_stddev_otsu=float(params.minOtsuStdDev),
    )

    max_border_errors = int(
        marker_size * marker_size * float(params.maxErroneousBitsInBorderRate)
    )
    detections: list[dict[str, Any]] = []
    for quad_index, (quad, corners, candidate_bits) in enumerate(
        zip(fitted_quads, corners_batch, candidate_bits_batch, strict=False)
    ):
        decoded: dict[str, Any] | None = None
        border_errors = _get_border_errors(
            candidate_bits,
            marker_size=marker_size,
            border_size=marker_border_bits,
        )
        is_inverted = False
        if bool(params.detectInvertedMarker):
            inverted = (1 - candidate_bits).astype(np.uint8, copy=False)
            inverted_errors = _get_border_errors(
                inverted,
                marker_size=marker_size,
                border_size=marker_border_bits,
            )
            if inverted_errors < border_errors:
                candidate_bits = inverted
                border_errors = inverted_errors
                is_inverted = True

        if border_errors <= max_border_errors:
            inner_bits = candidate_bits[
                marker_border_bits : marker_border_bits + marker_size,
                marker_border_bits : marker_border_bits + marker_size,
            ]
            identified, marker_id, rotation = dictionary.identify(
                inner_bits,
                float(params.errorCorrectionRate),
            )
            if identified:
                corrected_corners = np.roll(corners.astype(np.float64), int(rotation), axis=0)
                decoded = {
                    "id": int(marker_id),
                    "rotation": int(rotation),
                    "corners": corrected_corners,
                    "corners_raw": corners.astype(np.float64),
                    "is_inverted": is_inverted,
                    "border_errors": int(border_errors),
                    "quad_index": quad_index,
                }

        if decoded is None and cpu_fallback_on_miss:
            decoded = decode_single_quad_opencv(
                image_gray=image_gray,
                corners=corners,
                dictionary=dictionary,
                detector_params=params,
            )
            if decoded is not None:
                decoded["quad_index"] = quad_index
                decoded["decode_backend"] = "cpu_fallback"

        if decoded is None:
            continue

        decoded.setdefault("decode_backend", "gpu")
        if "blob_index" in quad:
            decoded["blob_index"] = int(quad["blob_index"])
        if "reversed_border" in quad:
            decoded["reversed_border"] = bool(quad["reversed_border"])
        if "score" in quad:
            decoded["score"] = float(quad["score"])
        detections.append(decoded)

    return detections
