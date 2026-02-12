import cv2
import slangpy as spy
import pathlib
import numpy as np
import sys

try:
    from slangtag.decode_tags import decode_tags_from_fitted_quads
except ImportError:
    from decode_tags import decode_tags_from_fitted_quads


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()



def crop_image_to_multiple(image: np.ndarray, multiple: int) -> np.ndarray:
    height, width = image.shape[:2]
    cropped_height = height - (height % multiple)
    cropped_width = width - (width % multiple)
    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(
            f"input image {width}x{height} is too small for {multiple}x alignment"
        )

    if cropped_height != height or cropped_width != width:
        print(
            "cropping input image from "
            f"{width}x{height} to {cropped_width}x{cropped_height} "
            f"for {multiple}x alignment"
        )

    return image[:cropped_height, :cropped_width]

def visualize_blob_extents(device, out_tex, blob_extents: np.ndarray, name_suffix: str = "") -> None:
    height = out_tex.height
    width = out_tex.width

    # Coverage map of how many extents touch each pixel.
    extent_coverage = np.zeros((height, width), dtype=np.uint16)

    thresholded = out_tex.to_numpy()
    if thresholded.ndim == 3:
        thresholded = thresholded[:, :, 0]
    thresholded = np.ascontiguousarray(thresholded.astype(np.uint8))

    overlay = np.stack([thresholded, thresholded, thresholded], axis=-1)

    for i, extent in enumerate(blob_extents):
        min_x = int(np.clip(extent[2], 0, width - 1))
        max_x = int(np.clip(extent[3], 0, width - 1))
        min_y = int(np.clip(extent[4], 0, height - 1))
        max_y = int(np.clip(extent[5], 0, height - 1))

        if min_x > max_x or min_y > max_y:
            continue

        extent_coverage[min_y, min_x:max_x + 1] += 1
        extent_coverage[max_y, min_x:max_x + 1] += 1
        extent_coverage[min_y:max_y + 1, min_x] += 1
        extent_coverage[min_y:max_y + 1, max_x] += 1

        color = np.array(
            [(37 * i) % 255 + 1, (67 * i) % 255 + 1, (97 * i) % 255 + 1],
            dtype=np.uint8,
        )
        overlay[min_y, min_x:max_x + 1] = color
        overlay[max_y, min_x:max_x + 1] = color
        overlay[min_y:max_y + 1, min_x] = color
        overlay[min_y:max_y + 1, max_x] = color

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        overlay[center_y, center_x] = np.array([255, 255, 255], dtype=np.uint8)

    if np.max(extent_coverage) > 0:
        extent_coverage_u8 = np.ascontiguousarray(
            (extent_coverage * (255.0 / np.max(extent_coverage))).astype(np.uint8)
        )
    else:
        extent_coverage_u8 = np.zeros((height, width), dtype=np.uint8)

    extent_coverage_tex = device.create_texture(
        width=width,
        height=height,
        format=spy.Format.r8_uint,
        usage=spy.TextureUsage.shader_resource,
        data=extent_coverage_u8,
    )
    spy.tev.show(extent_coverage_tex, name=f"blob extent coverage{name_suffix}")
    spy.tev.show(
        bitmap=spy.Bitmap(np.ascontiguousarray(overlay), spy.Bitmap.PixelFormat.rgb),
        name=f"blob extents overlay{name_suffix}",
    )


FITTED_QUAD_WORDS_PER_QUAD = 15
FITTED_QUAD_BLOB_INDEX_WORD = 0
FITTED_QUAD_REVERSED_BORDER_WORD = 1
FITTED_QUAD_SCORE_WORD = 2
FITTED_QUAD_CORNERS_START_WORD = 3
FITTED_QUAD_CORNERS_END_WORD = 11
FITTED_QUAD_CORNER_INDICES_START_WORD = 11
FITTED_QUAD_CORNER_INDICES_END_WORD = 15


def decode_fitted_quads(fitted_quad_words: np.ndarray, fitted_quad_count: int):
    if fitted_quad_count <= 0:
        return []

    packed = np.ascontiguousarray(
        fitted_quad_words[: fitted_quad_count * FITTED_QUAD_WORDS_PER_QUAD]
    ).reshape((fitted_quad_count, FITTED_QUAD_WORDS_PER_QUAD))
    corner_words = np.ascontiguousarray(
        packed[:, FITTED_QUAD_CORNERS_START_WORD:FITTED_QUAD_CORNERS_END_WORD]
    )
    corners = corner_words.view(np.float32).astype(np.float64).reshape((fitted_quad_count, 4, 2))
    scores = packed[:, FITTED_QUAD_SCORE_WORD].view(np.float32).astype(np.float64)
    corner_indices = packed[
        :, FITTED_QUAD_CORNER_INDICES_START_WORD:FITTED_QUAD_CORNER_INDICES_END_WORD
    ].astype(np.int64)

    decoded = []
    for i in range(fitted_quad_count):
        decoded.append(
            {
                "blob_index": int(packed[i, FITTED_QUAD_BLOB_INDEX_WORD]),
                "reversed_border": bool(packed[i, FITTED_QUAD_REVERSED_BORDER_WORD] != 0),
                "corners": corners[i].copy(),
                "corner_indices": corner_indices[i].tolist(),
                "score": float(scores[i]),
            }
        )
    return decoded


def visualize_quads(image_gray: np.ndarray, quads, name: str) -> None:
    overlay = np.stack([image_gray, image_gray, image_gray], axis=-1).astype(np.uint8)
    height, width = image_gray.shape
    for i, quad in enumerate(quads):
        corners = np.round(quad["corners"]).astype(np.int32)
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
        color = (
            int((37 * i) % 255 + 1),
            int((67 * i) % 255 + 1),
            int((97 * i) % 255 + 1),
        )
        cv2.polylines(overlay, [corners.reshape((-1, 1, 2))], True, color, 2, cv2.LINE_AA)
    spy.tev.show(
        bitmap=spy.Bitmap(np.ascontiguousarray(overlay), spy.Bitmap.PixelFormat.rgb),
        name=name,
    )


def visualize_decoded_tags(image_gray: np.ndarray, detections, name: str) -> None:
    overlay = np.stack([image_gray, image_gray, image_gray], axis=-1).astype(np.uint8)
    height, width = image_gray.shape
    for i, detection in enumerate(detections):
        corners = np.round(detection["corners"]).astype(np.int32)
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)

        color = (
            int((29 * i) % 255 + 1),
            int((89 * i) % 255 + 1),
            int((149 * i) % 255 + 1),
        )
        cv2.polylines(overlay, [corners.reshape((-1, 1, 2))], True, color, 2, cv2.LINE_AA)

        label = str(detection["id"])
        anchor = tuple(corners[0].tolist())
        cv2.putText(
            overlay,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    spy.tev.show(
        bitmap=spy.Bitmap(np.ascontiguousarray(overlay), spy.Bitmap.PixelFormat.rgb),
        name=name,
    )


device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ],
    type=spy.DeviceType.vulkan,
    enable_print=True,
    enable_debug_layers=True,
)

# Load the module
module = spy.Module.load_from_file(device, "shaders/threshold.slang")

img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = crop_image_to_multiple(img, 8)

tex = device.create_texture(
    width=img.shape[1],
    height=img.shape[0],
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource,
    data=np.ascontiguousarray(img.astype(np.uint8))
)

decimated_tex = device.create_texture(
    width=tex.width // 2,
    height=tex.height // 2,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)
decimate = tex.width // decimated_tex.width

unfiltered_minmax_tex = device.create_texture(
    width=decimated_tex.width // 4,
    height=decimated_tex.height // 4,
    format=spy.Format.rg8_uint,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

minmax_tex = device.create_texture(
    width=decimated_tex.width // 4,
    height=decimated_tex.height // 4,
    format=spy.Format.rg8_uint,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

out_tex = device.create_texture(
    width=tex.width // 2,
    height=tex.height // 2,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

# Display it with tev
spy.tev.show(tex, name="photo")


module.decimate(
    spy.grid(
        shape=(
            tex.height,
            tex.width,
        )
    ),
    tex,
    decimated_tex,
)


# Display the result
spy.tev.show(decimated_tex, name="decimated")

module.minmax(
    spy.grid(
        shape=(
            unfiltered_minmax_tex.height,
            unfiltered_minmax_tex.width,
        )
    ),
    decimated_tex,
    unfiltered_minmax_tex,
)

# Show r and g channels of minmax texture separately
unfiltered_minmax_tex_r = device.create_texture(
    width=unfiltered_minmax_tex.width,
    height=unfiltered_minmax_tex.height,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource,
    data=np.ascontiguousarray(unfiltered_minmax_tex.to_numpy()[:, :, 0])
)

unfiltered_minmax_tex_g = device.create_texture(
    width=unfiltered_minmax_tex.width,
    height=unfiltered_minmax_tex.height,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource,
    data=np.ascontiguousarray(unfiltered_minmax_tex.to_numpy()[:, :, 1]),
)
spy.tev.show(unfiltered_minmax_tex_r, name="unfiltered min")
spy.tev.show(unfiltered_minmax_tex_g, name="unfiltered max")

module.filter_minmax(
    spy.grid(
        shape=(
            unfiltered_minmax_tex.height,
            unfiltered_minmax_tex.width,
        )
    ),
    unfiltered_minmax_tex,
    minmax_tex,
)

# Show the filtered minmax texture
minmax_tex_r = device.create_texture(
    width=minmax_tex.width,
    height=minmax_tex.height,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource,
    data=np.ascontiguousarray(minmax_tex.to_numpy()[:, :, 0])
)

minmax_tex_g = device.create_texture(
    width=minmax_tex.width,
    height=minmax_tex.height,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource,
    data=np.ascontiguousarray(minmax_tex.to_numpy()[:, :, 1]),
)
spy.tev.show(minmax_tex_r, name="min")
spy.tev.show(minmax_tex_g, name="max")

module.threshold(
    spy.grid(
        shape=(
            decimated_tex.height,
            decimated_tex.width,
        )
    ),
    decimated_tex,
    minmax_tex,
    out_tex,
    25
)

# Show the output texture
spy.tev.show(out_tex, name="thresholded")

ccl_module = spy.Module.load_from_file(device, "shaders/ccl.slang")

ccl_out_buf = device.create_buffer(
    size=out_tex.width * out_tex.height * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

ccl_module.init(
    spy.grid(
        shape=(
            out_tex.height // 2,
            out_tex.width // 2,
        )
    ),
    out_tex,
    ccl_out_buf,
)

spy.tev.show(bitmap=spy.Bitmap(ccl_out_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.float32), spy.Bitmap.PixelFormat.r), name="ccl init")

ccl_module.compression(
    spy.grid(
        shape=(
            out_tex.height // 2,
            out_tex.width // 2,
        )
    ),
    ccl_out_buf,
    spy.uint2(out_tex.width, out_tex.height),
)
spy.tev.show(bitmap=spy.Bitmap(ccl_out_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.float32), spy.Bitmap.PixelFormat.r), name="ccl compress")


ccl_module.merge(
    spy.grid(
        shape=(
            out_tex.height // 2,
            out_tex.width // 2,
        )
    ),
    ccl_out_buf,
    spy.uint2(out_tex.width, out_tex.height),
)
spy.tev.show(bitmap=spy.Bitmap(ccl_out_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.float32), spy.Bitmap.PixelFormat.r), name="ccl merge")

ccl_module.compression(
    spy.grid(
        shape=(
            out_tex.height // 2,
            out_tex.width // 2,
        )
    ),
    ccl_out_buf,
    spy.uint2(out_tex.width, out_tex.height),
)
spy.tev.show(bitmap=spy.Bitmap(ccl_out_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.float32), spy.Bitmap.PixelFormat.r), name="ccl compress 2")


union_markers_size_buf = device.create_buffer(
    size=out_tex.width * out_tex.height * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

ccl_module.final_labeling(
    spy.grid(
        shape=(
            out_tex.height // 2,
            out_tex.width // 2,
        )
    ),
    ccl_out_buf,
    union_markers_size_buf,
    spy.uint2(out_tex.width, out_tex.height),
)
spy.tev.show(bitmap=spy.Bitmap(ccl_out_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.float32), spy.Bitmap.PixelFormat.r), name="ccl final labeling")
spy.tev.show(bitmap=spy.Bitmap(union_markers_size_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.float32), spy.Bitmap.PixelFormat.r), name="union markers size")

blob_diff_words_per_point = 6
blob_diff_points_per_offset = (out_tex.width - 2) * (out_tex.height - 2)
blob_diff_total_points = blob_diff_points_per_offset * 4
blob_diff_out_buf = device.create_buffer(
    size=blob_diff_total_points * blob_diff_words_per_point * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

blob_module = spy.Module.load_from_file(device, "shaders/blob.slang")
select_module = spy.Module.load_from_file(device, "shaders/select.slang")
sort_module = spy.Module.load_from_file(device, "shaders/sort.slang")
filter_module = spy.Module.load_from_file(device, "shaders/filter.slang")

blob_module.blob_diff(
    spy.grid(
        shape=(
            out_tex.height,
            out_tex.width,
        )
    ),
    out_tex,
    ccl_out_buf,
    union_markers_size_buf,
    blob_diff_out_buf,
    spy.uint2(out_tex.width, out_tex.height),
    25,
)

blob_diff_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)

select_module.count_nonzero_blob_diff_points(
    spy.grid(
        shape=(
            blob_diff_total_points,
        )
    ),
    blob_diff_out_buf,
    blob_diff_count_buf,
    blob_diff_total_points,
)

blob_diff_compacted_size = int(blob_diff_count_buf.to_numpy()[0])

blob_diff_compacted_buf = device.create_buffer(
    size=max(1, blob_diff_compacted_size) * blob_diff_words_per_point * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

blob_diff_filter_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)

select_module.filter_nonzero_blob_diff_points(
    spy.grid(
        shape=(
            blob_diff_total_points,
        )
    ),
    blob_diff_out_buf,
    blob_diff_compacted_buf,
    blob_diff_filter_count_buf,
    blob_diff_total_points,
)

blob_diff_filtered_size = int(blob_diff_filter_count_buf.to_numpy()[0])
print(
    f"blob_diff compacted points: {blob_diff_filtered_size}/{blob_diff_total_points}"
)

blob_diff = blob_diff_out_buf.to_numpy().reshape((4, blob_diff_points_per_offset, blob_diff_words_per_point))
blob_diff_nonzero = int(np.count_nonzero(blob_diff[:, :, 1]))
print(f"blob_diff nonzero points: {blob_diff_nonzero}/{blob_diff_total_points}")

blob_diff_compacted = blob_diff_compacted_buf.to_numpy()[
    : blob_diff_filtered_size * blob_diff_words_per_point
].reshape((blob_diff_filtered_size, blob_diff_words_per_point))
if blob_diff_filtered_size > 0:
    print(f"first compacted point: {blob_diff_compacted[0]}")

blob_diff_sort_points = next_power_of_two(blob_diff_filtered_size)
blob_diff_sorted_buf = device.create_buffer(
    size=max(1, blob_diff_sort_points) * blob_diff_words_per_point * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

sort_module.prepare_blob_diff_points(
    spy.grid(
        shape=(
            blob_diff_sort_points,
        )
    ),
    blob_diff_compacted_buf,
    blob_diff_sorted_buf,
    blob_diff_filtered_size,
    blob_diff_sort_points,
)

k = 2
while k <= blob_diff_sort_points:
    j = k // 2
    while j > 0:
        sort_module.bitonic_sort_blob_diff_points(
            spy.grid(
                shape=(
                    blob_diff_sort_points,
                )
            ),
            blob_diff_sorted_buf,
            blob_diff_sort_points,
            j,
            k,
        )
        j //= 2
    k *= 2

blob_diff_sorted = blob_diff_sorted_buf.to_numpy()[
    : blob_diff_filtered_size * blob_diff_words_per_point
].reshape((blob_diff_filtered_size, blob_diff_words_per_point))
if blob_diff_filtered_size > 0:
    blob_diff_sorted_keys = (
        blob_diff_sorted[:, 1].astype(np.uint64) << 20
    ) | blob_diff_sorted[:, 0].astype(np.uint64)
    print(f"first sorted point: {blob_diff_sorted[0]}")
    print(f"sorted key order valid: {bool(np.all(blob_diff_sorted_keys[:-1] <= blob_diff_sorted_keys[1:]))}")

blob_extent_words_per_extent = 11
blob_extent_buf = device.create_buffer(
    size=max(1, blob_diff_filtered_size) * blob_extent_words_per_extent * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
blob_extent_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)

filter_module.build_blob_pair_extents(
    spy.grid(shape=(1,)),
    blob_diff_sorted_buf,
    blob_extent_buf,
    blob_extent_count_buf,
    blob_diff_filtered_size,
)

blob_extent_count = int(blob_extent_count_buf.to_numpy()[0])
print(f"blob pair extents: {blob_extent_count}")
if blob_extent_count > 0:
    blob_extents = blob_extent_buf.to_numpy()[
        : blob_extent_count * blob_extent_words_per_extent
    ].reshape((blob_extent_count, blob_extent_words_per_extent))
    print(f"first extent: {blob_extents[0]}")
    visualize_blob_extents(device, out_tex, blob_extents)

filtered_blob_extent_buf = device.create_buffer(
    size=max(1, blob_extent_count) * blob_extent_words_per_extent * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
selected_blob_extent_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)
selected_blob_point_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)

min_tag_width = 3
reversed_border = 1
normal_border = 1
min_cluster_pixels = 24
max_cluster_pixels = 4 * (out_tex.width + out_tex.height)

filter_module.filter_blob_pair_extents(
    spy.grid(shape=(1,)),
    blob_extent_buf,
    filtered_blob_extent_buf,
    selected_blob_extent_count_buf,
    selected_blob_point_count_buf,
    blob_extent_count,
    min_tag_width,
    reversed_border,
    normal_border,
    min_cluster_pixels,
    max_cluster_pixels,
)

selected_blob_extent_count = int(selected_blob_extent_count_buf.to_numpy()[0])
selected_blob_point_count = int(selected_blob_point_count_buf.to_numpy()[0])
print(f"selected blob pair extents: {selected_blob_extent_count}/{blob_extent_count}")
print(f"selected blob points: {selected_blob_point_count}/{blob_diff_filtered_size}")

filtered_blob_extents = np.zeros((0, blob_extent_words_per_extent), dtype=np.uint32)
if blob_extent_count > 0:
    filtered_blob_extents = filtered_blob_extent_buf.to_numpy()[
        : blob_extent_count * blob_extent_words_per_extent
    ].reshape((blob_extent_count, blob_extent_words_per_extent))
    passing_blob_extents = filtered_blob_extents[
        filtered_blob_extents[:, 7] > 0
    ]
    if passing_blob_extents.shape[0] > 0:
        visualize_blob_extents(device, out_tex, passing_blob_extents, " filtered")

selected_blob_point_words_per_point = 4
selected_blob_points_buf = device.create_buffer(
    size=max(1, selected_blob_point_count) * selected_blob_point_words_per_point * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

filter_module.rewrite_selected_blob_points_with_theta(
    spy.grid(
        shape=(
            blob_diff_filtered_size,
        )
    ),
    blob_diff_sorted_buf,
    blob_extent_buf,
    filtered_blob_extent_buf,
    selected_blob_points_buf,
    blob_extent_count,
    blob_diff_filtered_size,
)

selected_blob_points = selected_blob_points_buf.to_numpy()[
    : selected_blob_point_count * selected_blob_point_words_per_point
].reshape((selected_blob_point_count, selected_blob_point_words_per_point))
if selected_blob_point_count > 0:
    print(f"first selected blob point (unsorted): {selected_blob_points[0]}")

selected_blob_sort_points = next_power_of_two(selected_blob_point_count)
selected_blob_sorted_points_buf = device.create_buffer(
    size=max(1, selected_blob_sort_points) * selected_blob_point_words_per_point * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

sort_module.prepare_selected_blob_points(
    spy.grid(
        shape=(
            selected_blob_sort_points,
        )
    ),
    selected_blob_points_buf,
    selected_blob_sorted_points_buf,
    selected_blob_point_count,
    selected_blob_sort_points,
)

k = 2
while k <= selected_blob_sort_points:
    j = k // 2
    while j > 0:
        sort_module.bitonic_sort_selected_blob_points(
            spy.grid(
                shape=(
                    selected_blob_sort_points,
                )
            ),
            selected_blob_sorted_points_buf,
            selected_blob_sort_points,
            j,
            k,
        )
        j //= 2
    k *= 2

selected_blob_points_sorted = selected_blob_sorted_points_buf.to_numpy()[
    : selected_blob_point_count * selected_blob_point_words_per_point
].reshape((selected_blob_point_count, selected_blob_point_words_per_point))
if selected_blob_point_count > 0:
    selected_blob_sort_keys = (
        selected_blob_points_sorted[:, 0].astype(np.uint64) << 32
    ) | selected_blob_points_sorted[:, 1].astype(np.uint64)
    print(f"first selected blob point (sorted): {selected_blob_points_sorted[0]}")
    print(
        "selected blob point key order valid: "
        f"{bool(np.all(selected_blob_sort_keys[:-1] <= selected_blob_sort_keys[1:]))}"
    )

line_fit_point_words_per_point = 10
line_fit_points_buf = device.create_buffer(
    size=max(1, selected_blob_point_count) * line_fit_point_words_per_point * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

filter_module.build_line_fit_points(
    spy.grid(shape=(1,)),
    selected_blob_sorted_points_buf,
    decimated_tex,
    line_fit_points_buf,
    selected_blob_point_count,
    decimated_tex.width,
    decimated_tex.height,
    decimate,
)

line_fit_points = line_fit_points_buf.to_numpy()[
    : selected_blob_point_count * line_fit_point_words_per_point
].reshape((selected_blob_point_count, line_fit_point_words_per_point))
if selected_blob_point_count > 0:
    print(f"first line fit point (prefix moments): {line_fit_points[0]}")

errs_buf = device.create_buffer(
    size=max(1, selected_blob_point_count) * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
filtered_errs_buf = device.create_buffer(
    size=max(1, selected_blob_point_count) * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
peak_words_per_peak = 3
peaks_buf = device.create_buffer(
    size=max(1, selected_blob_point_count) * peak_words_per_peak * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

filter_module.fit_line_errors_and_peaks(
    spy.grid(shape=(1,)),
    line_fit_points_buf,
    filtered_blob_extent_buf,
    errs_buf,
    filtered_errs_buf,
    peaks_buf,
    blob_extent_count,
    selected_blob_point_count,
)

if selected_blob_point_count > 0:
    errs = errs_buf.to_numpy()[:selected_blob_point_count].view(np.float32)
    filtered_errs = filtered_errs_buf.to_numpy()[:selected_blob_point_count].view(np.float32)
    print(f"line fit error range: [{float(np.min(errs)):.6f}, {float(np.max(errs)):.6f}]")
    print(
        "filtered line fit error range: "
        f"[{float(np.min(filtered_errs)):.6f}, {float(np.max(filtered_errs)):.6f}]"
    )

peak_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)
if selected_blob_point_count > 0:
    select_module.count_valid_peaks(
        spy.grid(shape=(selected_blob_point_count,)),
        peaks_buf,
        peak_count_buf,
        selected_blob_point_count,
    )

peak_count = int(peak_count_buf.to_numpy()[0])
print(f"valid peaks: {peak_count}/{selected_blob_point_count}")

compacted_peaks_buf = device.create_buffer(
    size=max(1, peak_count) * peak_words_per_peak * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
compacted_peak_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)
if selected_blob_point_count > 0:
    select_module.filter_valid_peaks(
        spy.grid(shape=(selected_blob_point_count,)),
        peaks_buf,
        compacted_peaks_buf,
        compacted_peak_count_buf,
        selected_blob_point_count,
    )

compacted_peak_count = int(compacted_peak_count_buf.to_numpy()[0])
print(f"compacted peaks: {compacted_peak_count}")

peak_sort_points = next_power_of_two(compacted_peak_count)
sorted_peaks_buf = device.create_buffer(
    size=max(1, peak_sort_points) * peak_words_per_peak * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

sort_module.prepare_peaks(
    spy.grid(shape=(peak_sort_points,)),
    compacted_peaks_buf,
    sorted_peaks_buf,
    compacted_peak_count,
    peak_sort_points,
)

k = 2
while k <= peak_sort_points:
    j = k // 2
    while j > 0:
        sort_module.bitonic_sort_peaks(
            spy.grid(shape=(peak_sort_points,)),
            sorted_peaks_buf,
            peak_sort_points,
            j,
            k,
        )
        j //= 2
    k *= 2

sorted_peaks = np.zeros((0, peak_words_per_peak), dtype=np.uint32)
if compacted_peak_count > 0:
    sorted_peaks = sorted_peaks_buf.to_numpy()[
        : compacted_peak_count * peak_words_per_peak
    ].reshape((compacted_peak_count, peak_words_per_peak))
    print(f"first sorted peak: {sorted_peaks[0]}")

peak_extent_words_per_extent = 3
peak_extent_buf = device.create_buffer(
    size=max(1, compacted_peak_count) * peak_extent_words_per_extent * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
peak_extent_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)

filter_module.build_peak_extents(
    spy.grid(shape=(1,)),
    sorted_peaks_buf,
    peak_extent_buf,
    peak_extent_count_buf,
    compacted_peak_count,
)

peak_extent_count = int(peak_extent_count_buf.to_numpy()[0])
print(f"peak extents: {peak_extent_count}")
peak_extents = np.zeros((0, peak_extent_words_per_extent), dtype=np.uint32)
if peak_extent_count > 0:
    peak_extents = peak_extent_buf.to_numpy()[
        : peak_extent_count * peak_extent_words_per_extent
    ].reshape((peak_extent_count, peak_extent_words_per_extent))
    print(f"first peak extent: {peak_extents[0]}")

max_nmaxima = 10
max_line_fit_mse = 10.0
cos_critical_rad = 0.984807753012208
fitted_quads_buf = device.create_buffer(
    size=max(1, peak_extent_count) * FITTED_QUAD_WORDS_PER_QUAD * 4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
fitted_quad_count_buf = device.create_buffer(
    size=4,
    format=spy.Format.r32_uint,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    data=np.zeros(1, dtype=np.uint32),
)

filter_module.fit_quads(
    spy.grid(shape=(max(1, peak_extent_count),)),
    sorted_peaks_buf,
    peak_extent_buf,
    line_fit_points_buf,
    filtered_blob_extent_buf,
    fitted_quads_buf,
    fitted_quad_count_buf,
    peak_extent_count,
    blob_extent_count,
    max_nmaxima,
    max_line_fit_mse,
    cos_critical_rad,
    min_tag_width,
    float(decimate),
)

fitted_quad_count = int(fitted_quad_count_buf.to_numpy()[0])
fitted_quads = decode_fitted_quads(fitted_quads_buf.to_numpy(), fitted_quad_count)
print(f"fitted quads (shader): {len(fitted_quads)}")
if fitted_quads:
    print(f"first fitted quad: {fitted_quads[0]}")
    visualize_quads(img, fitted_quads, name="fitted quads")

tag_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
tag_params = cv2.aruco.DetectorParameters()
tag_params.detectInvertedMarker = True
decoded_tags = decode_tags_from_fitted_quads(
    image_gray=img,
    fitted_quads=fitted_quads,
    dictionary=tag_dictionary,
    detector_params=tag_params,
)
print(f"decoded tags: {len(decoded_tags)}")
if decoded_tags:
    print(f"first decoded tag: {decoded_tags[0]}")
    visualize_decoded_tags(img, decoded_tags, name="decoded tags")

blob_diff_density = np.count_nonzero(blob_diff[:, :, 1], axis=0).reshape((out_tex.height - 2, out_tex.width - 2))
blob_diff_density_img = np.ascontiguousarray((blob_diff_density * 85).astype(np.uint8))
blob_diff_density_tex = device.create_texture(
    width=out_tex.width - 2,
    height=out_tex.height - 2,
    format=spy.Format.r8_uint,
    usage=spy.TextureUsage.shader_resource,
    data=blob_diff_density_img,
)
spy.tev.show(blob_diff_density_tex, name="blob diff density")

out_ccl = ccl_out_buf.to_numpy().reshape((out_tex.height, out_tex.width)).astype(np.uint32)
num_unique_labels = len(np.unique(out_ccl))

# Create a color for each label and display the result
# Keep in mind that the labels are not necessarily contiguous, so we need to create a mapping from label to color
unique_labels = np.unique(out_ccl)
label_to_color = {label: np.random.randint(0, 255, size=3) for label in unique_labels}
color_ccl = np.zeros((out_tex.height, out_tex.width, 3), dtype=np.uint8)
for label, color in label_to_color.items():
    color_ccl[out_ccl == label] = color
spy.tev.show(bitmap=spy.Bitmap(color_ccl, spy.Bitmap.PixelFormat.rgb), name="ccl colored")
