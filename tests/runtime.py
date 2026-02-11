import os
import pathlib

import numpy as np
import pytest

try:
    import slangpy as spy
except ImportError:  # pragma: no cover - exercised when slangpy is unavailable
    spy = None


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SHADER_ROOT = _REPO_ROOT / "slangtag"


class Runtime:
    def __init__(self):
        self.available = False
        self.device_name = None
        self.device = None
        self.ccl_module = None
        self.blob_module = None
        self.select_module = None
        self.filter_module = None
        self.sort_module = None
        self.supports_compression = False
        self.supports_init = False
        self.supports_blob_diff = False
        self.supports_select = False
        self.supports_filter = False
        self.supports_sort = False
        self.skip_reason = "slangpy is not installed"

    def probe(self):
        if spy is None:
            return

        preferred = ["vulkan", "cpu"]
        env_choice = os.environ.get("SLANGTAG_TEST_DEVICE")
        candidates = [env_choice] if env_choice else preferred

        for name in candidates:
            if name is None:
                continue
            if not hasattr(spy.DeviceType, name):
                continue

            try:
                device = spy.create_device(
                    include_paths=[_SHADER_ROOT],
                    type=getattr(spy.DeviceType, name),
                )
                ccl_module = spy.Module.load_from_file(device, "shaders/ccl.slang")
                blob_module = spy.Module.load_from_file(device, "shaders/blob.slang")
                select_module = spy.Module.load_from_file(device, "shaders/select.slang")
                filter_module = spy.Module.load_from_file(device, "shaders/filter.slang")
                sort_module = spy.Module.load_from_file(device, "shaders/sort.slang")
            except Exception:
                continue

            self.device_name = name
            self.device = device
            self.ccl_module = ccl_module
            self.blob_module = blob_module
            self.select_module = select_module
            self.filter_module = filter_module
            self.sort_module = sort_module
            self.available = True
            self.skip_reason = ""
            break

        if not self.available:
            self.skip_reason = (
                "unable to create a SlangPy device/module; set "
                "SLANGTAG_TEST_DEVICE to a supported backend"
            )
            return

        self.supports_compression = self._probe_compression()
        self.supports_init = self._probe_init()
        self.supports_blob_diff = self._probe_blob_diff()
        self.supports_select = self._probe_select()
        self.supports_filter = self._probe_filter()
        self.supports_sort = self._probe_sort()

    def _probe_compression(self):
        labels = self.device.create_buffer(
            size=16 * 4,
            format=spy.Format.r32_uint,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=np.arange(16, dtype=np.uint32),
        )
        try:
            self.ccl_module.compression(spy.grid(shape=(2, 2)), labels, spy.uint2(4, 4))
            return True
        except Exception:
            return False

    def _probe_init(self):
        try:
            tex = self.device.create_texture(
                width=2,
                height=2,
                format=spy.Format.r8_uint,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                data=np.array([[255, 255], [255, 255]], dtype=np.uint8),
            )
        except Exception:
            return False

        labels = self.device.create_buffer(
            size=2 * 2 * 4,
            format=spy.Format.r32_uint,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

        try:
            self.ccl_module.init(spy.grid(shape=(1, 1)), tex, labels)
            return True
        except Exception:
            return False

    def _probe_blob_diff(self):
        width = 4
        height = 4
        points_per_offset = (width - 2) * (height - 2)
        words_per_point = 6

        try:
            tex = self.device.create_texture(
                width=width,
                height=height,
                format=spy.Format.r8_uint,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                data=np.zeros((height, width), dtype=np.uint8),
            )
            blobs = self.device.create_buffer(
                size=width * height * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(width * height, dtype=np.uint32),
            )
            union_sizes = self.device.create_buffer(
                size=width * height * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.full(width * height, 100, dtype=np.uint32),
            )
            result = self.device.create_buffer(
                size=points_per_offset * 4 * words_per_point * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            self.blob_module.blob_diff(
                spy.grid(shape=(height, width)),
                tex,
                blobs,
                union_sizes,
                result,
                spy.uint2(width, height),
                25,
            )
            return True
        except Exception:
            return False

    def _probe_select(self):
        words_per_point = 6
        total_points = 4
        try:
            input_data = np.zeros(total_points * words_per_point, dtype=np.uint32)
            input_data[1] = 10
            input_data[7] = 0
            input_data[13] = 22
            input_data[19] = 0

            input_buf = self.device.create_buffer(
                size=input_data.size * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=input_data,
            )
            count_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(1, dtype=np.uint32),
            )

            self.select_module.count_nonzero_blob_diff_points(
                spy.grid(shape=(total_points,)),
                input_buf,
                count_buf,
                total_points,
            )

            peaks = np.array(
                [
                    1,
                    np.array([-1.0], dtype=np.float32).view(np.uint32)[0],
                    0,
                    0xFFFF,
                    np.array([0.0], dtype=np.float32).view(np.uint32)[0],
                    1,
                ],
                dtype=np.uint32,
            )
            peaks_buf = self.device.create_buffer(
                size=peaks.size * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=peaks,
            )
            peak_count_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(1, dtype=np.uint32),
            )
            self.select_module.count_valid_peaks(
                spy.grid(shape=(2,)),
                peaks_buf,
                peak_count_buf,
                2,
            )
            return True
        except Exception:
            return False

    def _probe_filter(self):
        words_per_extent = 11
        valid_points = 1
        try:
            input_data = np.array([3, 7, 10, 20, 0, 1], dtype=np.uint32)
            input_buf = self.device.create_buffer(
                size=input_data.size * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=input_data,
            )
            extents_buf = self.device.create_buffer(
                size=words_per_extent * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            extent_count_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(1, dtype=np.uint32),
            )
            self.filter_module.build_blob_pair_extents(
                spy.grid(shape=(1,)),
                input_buf,
                extents_buf,
                extent_count_buf,
                valid_points,
            )

            filtered_extents_buf = self.device.create_buffer(
                size=words_per_extent * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            selected_extent_count_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(1, dtype=np.uint32),
            )
            selected_point_count_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(1, dtype=np.uint32),
            )

            self.filter_module.filter_blob_pair_extents(
                spy.grid(shape=(1,)),
                extents_buf,
                filtered_extents_buf,
                selected_extent_count_buf,
                selected_point_count_buf,
                1,
                3,
                1,
                1,
                24,
                100,
            )

            selected_points_buf = self.device.create_buffer(
                size=4 * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            self.filter_module.rewrite_selected_blob_points_with_theta(
                spy.grid(shape=(valid_points,)),
                input_buf,
                extents_buf,
                filtered_extents_buf,
                selected_points_buf,
                1,
                valid_points,
            )

            selected_sorted_points = np.array([0, 0, 2, 2], dtype=np.uint32)
            selected_sorted_points_buf = self.device.create_buffer(
                size=selected_sorted_points.size * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=selected_sorted_points,
            )
            decimated_tex = self.device.create_texture(
                width=4,
                height=4,
                format=spy.Format.r8_uint,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                data=np.zeros((4, 4), dtype=np.uint8),
            )
            line_fit_points_buf = self.device.create_buffer(
                size=10 * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            self.filter_module.build_line_fit_points(
                spy.grid(shape=(1,)),
                selected_sorted_points_buf,
                decimated_tex,
                line_fit_points_buf,
                1,
                4,
                4,
                2,
            )

            errs_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            filtered_errs_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            peaks_buf = self.device.create_buffer(
                size=3 * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            self.filter_module.fit_line_errors_and_peaks(
                spy.grid(shape=(1,)),
                line_fit_points_buf,
                filtered_extents_buf,
                errs_buf,
                filtered_errs_buf,
                peaks_buf,
                1,
                1,
            )

            peak_extents_buf = self.device.create_buffer(
                size=3 * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            peak_extent_count_buf = self.device.create_buffer(
                size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(1, dtype=np.uint32),
            )
            self.filter_module.build_peak_extents(
                spy.grid(shape=(1,)),
                peaks_buf,
                peak_extents_buf,
                peak_extent_count_buf,
                0,
            )
            return True
        except Exception:
            return False

    def _probe_sort(self):
        words_per_point = 4
        total_points = 2
        try:
            input_points = np.array(
                [
                    1,
                    2,
                    10,
                    10,
                    0,
                    3,
                    20,
                    20,
                ],
                dtype=np.uint32,
            )
            input_buf = self.device.create_buffer(
                size=input_points.size * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=input_points,
            )
            output_buf = self.device.create_buffer(
                size=total_points * words_per_point * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            self.sort_module.prepare_selected_blob_points(
                spy.grid(shape=(total_points,)),
                input_buf,
                output_buf,
                total_points,
                total_points,
            )
            self.sort_module.bitonic_sort_selected_blob_points(
                spy.grid(shape=(total_points,)),
                output_buf,
                total_points,
                1,
                2,
            )
            sorted_points = output_buf.to_numpy().reshape((total_points, words_per_point))
            expected_points = np.array(
                [
                    [0, 3, 20, 20],
                    [1, 2, 10, 10],
                ],
                dtype=np.uint32,
            )
            if not np.array_equal(sorted_points, expected_points):
                return False

            peak_input = np.array(
                [
                    1,
                    np.array([-0.5], dtype=np.float32).view(np.uint32)[0],
                    0,
                    0,
                    np.array([-1.0], dtype=np.float32).view(np.uint32)[0],
                    1,
                ],
                dtype=np.uint32,
            )
            peak_input_buf = self.device.create_buffer(
                size=peak_input.size * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=peak_input,
            )
            peak_output_buf = self.device.create_buffer(
                size=2 * 3 * 4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )
            self.sort_module.prepare_peaks(
                spy.grid(shape=(2,)),
                peak_input_buf,
                peak_output_buf,
                2,
                2,
            )
            self.sort_module.bitonic_sort_peaks(
                spy.grid(shape=(2,)),
                peak_output_buf,
                2,
                1,
                2,
            )
            sorted_peaks = peak_output_buf.to_numpy().reshape((2, 3))
            expected_peaks = np.array(
                [
                    [0, np.array([-1.0], dtype=np.float32).view(np.uint32)[0], 1],
                    [1, np.array([-0.5], dtype=np.float32).view(np.uint32)[0], 0],
                ],
                dtype=np.uint32,
            )
            if not np.array_equal(sorted_peaks, expected_peaks):
                return False
            return True
        except Exception:
            return False


RUNTIME = Runtime()
RUNTIME.probe()


def require_runtime():
    if not RUNTIME.available:
        pytest.skip(RUNTIME.skip_reason)


def require_support(feature_name):
    require_runtime()
    supported = getattr(RUNTIME, f"supports_{feature_name}")
    if not supported:
        pytest.skip(f"{feature_name} kernels are unsupported on backend {RUNTIME.device_name!r}")


def run_init(image):
    tex = RUNTIME.device.create_texture(
        width=image.shape[1],
        height=image.shape[0],
        format=spy.Format.r8_uint,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=np.ascontiguousarray(image),
    )

    labels = RUNTIME.device.create_buffer(
        size=image.size * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.ccl_module.init(
        spy.grid(shape=(image.shape[0] // 2, image.shape[1] // 2)),
        tex,
        labels,
    )
    return labels.to_numpy()


def run_blob_diff(image, blobs, union_sizes, min_blob_size):
    height, width = image.shape
    points_per_offset = (width - 2) * (height - 2)
    words_per_point = 6

    tex = RUNTIME.device.create_texture(
        width=width,
        height=height,
        format=spy.Format.r8_uint,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=np.ascontiguousarray(image),
    )
    blobs_buf = RUNTIME.device.create_buffer(
        size=image.size * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(blobs.reshape(-1)),
    )
    union_sizes_buf = RUNTIME.device.create_buffer(
        size=image.size * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.ascontiguousarray(union_sizes.reshape(-1)),
    )
    result_buf = RUNTIME.device.create_buffer(
        size=points_per_offset * 4 * words_per_point * 4,
        format=spy.Format.r32_uint,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    RUNTIME.blob_module.blob_diff(
        spy.grid(shape=(height, width)),
        tex,
        blobs_buf,
        union_sizes_buf,
        result_buf,
        spy.uint2(width, height),
        min_blob_size,
    )

    return result_buf.to_numpy().reshape((4, points_per_offset, words_per_point))
