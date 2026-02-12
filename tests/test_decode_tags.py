import cv2
import numpy as np

from slangtag.decode_tags import decode_single_quad_opencv, decode_tags_from_fitted_quads


def _make_marker_scene(
    marker_id: int,
    side_pixels: int = 160,
    canvas_size: int = 256,
    offset_xy: tuple[int, int] = (48, 48),
    invert: bool = False,
):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels, borderBits=1)
    if invert:
        marker = 255 - marker

    image = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    x, y = offset_xy
    image[y : y + side_pixels, x : x + side_pixels] = marker

    corners = np.array(
        [
            [x, y],
            [x + side_pixels - 1, y],
            [x + side_pixels - 1, y + side_pixels - 1],
            [x, y + side_pixels - 1],
        ],
        dtype=np.float32,
    )
    return image, corners, dictionary


def test_decode_single_quad_decodes_tag_id():
    marker_id = 23
    image, corners, dictionary = _make_marker_scene(marker_id=marker_id)

    decoded = decode_single_quad_opencv(
        image_gray=image,
        corners=corners,
        dictionary=dictionary,
    )

    assert decoded is not None
    assert decoded["id"] == marker_id


def test_decode_single_quad_corrects_corner_rotation():
    marker_id = 7
    image, corners, dictionary = _make_marker_scene(marker_id=marker_id)
    corners_rotated_start = np.roll(corners, -1, axis=0)

    decoded = decode_single_quad_opencv(
        image_gray=image,
        corners=corners_rotated_start,
        dictionary=dictionary,
    )

    assert decoded is not None
    np.testing.assert_allclose(decoded["corners"], corners.astype(np.float64), atol=1.0, rtol=0.0)


def test_decode_single_quad_handles_inverted_markers_when_enabled():
    marker_id = 5
    image, corners, dictionary = _make_marker_scene(marker_id=marker_id, invert=True)

    no_invert_params = cv2.aruco.DetectorParameters()
    no_invert_params.detectInvertedMarker = False
    rejected = decode_single_quad_opencv(
        image_gray=image,
        corners=corners,
        dictionary=dictionary,
        detector_params=no_invert_params,
    )
    assert rejected is None

    invert_params = cv2.aruco.DetectorParameters()
    invert_params.detectInvertedMarker = True
    decoded = decode_single_quad_opencv(
        image_gray=image,
        corners=corners,
        dictionary=dictionary,
        detector_params=invert_params,
    )
    assert decoded is not None
    assert decoded["id"] == marker_id
    assert decoded["is_inverted"] is True


def test_decode_tags_from_fitted_quads_preserves_quad_metadata():
    marker_id = 11
    image, corners, dictionary = _make_marker_scene(marker_id=marker_id)
    fitted_quads = [
        {
            "blob_index": 42,
            "reversed_border": False,
            "score": 0.125,
            "corners": corners.astype(np.float64),
        }
    ]

    decoded = decode_tags_from_fitted_quads(
        image_gray=image,
        fitted_quads=fitted_quads,
        dictionary=dictionary,
    )

    assert len(decoded) == 1
    assert decoded[0]["id"] == marker_id
    assert decoded[0]["blob_index"] == 42
    assert decoded[0]["quad_index"] == 0
