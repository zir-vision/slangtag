import cv2
import slangpy as spy
import pathlib
import numpy as np

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

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
            decimated_tex.height,
            decimated_tex.width,
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
    5
)

# Show the output texture
spy.tev.show(out_tex, name="thresholded")