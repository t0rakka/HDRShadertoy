# HDR Shadertoy

A Vulkan **HDR viewer** for [Shadertoy](https://www.shadertoy.com/) effects, converted to **compute shaders**.

Shaders render into a float16 target and resolve to an HDR (or SDR) swapchain — so bright highlights and emissive scenes can use the display’s real dynamic range instead of being crushed into 8-bit LDR.

Built on [mango](https://github.com/t0rakka/mango) as the foundation library (windowing, Vulkan, shader compilation, HDR path).

> **Work in progress.** Not all Shadertoy features are supported yet (buffers, multipass, textures, sound, etc.). The included effects are hand-ported `.comp` shaders that fit the current single-pass compute model.

## Controls

| Key | Action |
|-----|--------|
| **← / →** | Previous / next effect |
| **F** | Toggle fullscreen |
| **ESC** | Quit |

## Build

Requires a built and installed [mango](https://github.com/t0rakka/mango) (`find_package(mango)` / `mango::vulkan`), a C++20 toolchain, CMake 3.19+, and a Vulkan-capable GPU/driver.

```bash
git clone https://github.com/t0rakka/HDRShadertoy.git
cd HDRShadertoy
cmake -S . -B build -G Ninja
cmake --build build
./build/shadertoy
```

Optional flags: `--shader <file.comp>`, `--sdr`, `--info`, `--validate`, `--help`.
