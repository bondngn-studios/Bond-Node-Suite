# ComfyUI Manual Installation Guide
### For use with SimplePod — CUDA 13 Image

This guide walks you through setting up ComfyUI with all required custom nodes on a fresh SimplePod instance. It assumes you are comfortable with a terminal.

---

## Prerequisites

Before starting, you will need a SimplePod account and a persistent storage volume. The storage volume is where your models and custom nodes will live across sessions.

**Pod settings:**
- Image: `simplepodai/comfyui`, tag `cuda13.0`
- Mount point: `/app`
- Expose port `8188`, name it `ComfyUI`
- Persistent volume: attach your storage volume, mounted at `/app`

> If your workflows require heavy models and you get out-of-memory errors, add `--disable-smart-memory --disable-cuda-malloc` to the Docker entrypoint arguments field.

---

## Step 1 — Open a Terminal

Once your pod is running, open a terminal via Jupyter (port 8888) or SSH. All commands below are run as root.

---

## Step 2 — Install System Dependencies

Update apt and install exiftool (required for Bond Node Suite metadata features):

```bash
apt-get update
apt-get install -y libimage-exiftool-perl
```

---

## Step 3 — Fix pip Packages

The default image ships with versions of `flash-attn` and `bitsandbytes` that are incompatible with Blackwell GPUs (RTX 5000 series). Run these two commands regardless of your GPU — they are safe on all hardware:

```bash
python3.11 -m pip uninstall flash-attn -y
python3.11 -m pip install bitsandbytes==0.45.5
```

---

## Step 4 — Clone Custom Nodes

Navigate to the custom nodes directory and clone each node pack:

```bash
cd /app/ComfyUI/custom_nodes
```

Then clone each of the following:

```bash
git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite
git clone --depth=1 https://github.com/crystian/ComfyUI-Crystools.git ComfyUI-Crystools
git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git comfyui-kjnodes
git clone --depth=1 https://github.com/DoctorDiffusion/ComfyUI-MediaMixer.git ComfyUI-MediaMixer
git clone --depth=1 https://github.com/TinyTerra/ComfyUI_tinyterranodes.git comfyui_tinyterranodes
git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux.git comfyui_controlnet_aux
git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git comfyui-impact-pack
git clone --depth=1 https://github.com/1038lab/ComfyUI-QwenVL.git ComfyUI-QwenVL
git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use
git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git ComfyUI-WanVideoWrapper
git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git ComfyUI-GGUF
git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git comfyui-rgthree
git clone --depth=1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git ComfyUI-Custom-Scripts
git clone --depth=1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2_videoupscaler
git clone --depth=1 https://github.com/princepainter/ComfyUI-PainterI2V.git ComfyUI-PainterI2V
git clone --depth=1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git ComfyUI-Frame-Interpolation
git clone --depth=1 https://github.com/bondngn-studios/Bond-Node-Suite.git Bond-Node-Suite
```

---

## Step 5 — Install Node Requirements

Install dependencies for each node pack that has a `requirements.txt`:

```bash
cd /app/ComfyUI

for dir in custom_nodes/*/; do
    if [ -f "${dir}requirements.txt" ]; then
        echo "Installing requirements for ${dir}..."
        python3.11 -m pip install -r "${dir}requirements.txt"
    fi
done
```

This may take several minutes depending on your connection speed. You will see a lot of output — that is normal.

---

## Step 6 — Verify

Do a quick sanity check to confirm all node directories are present:

```bash
ls /app/ComfyUI/custom_nodes/
```

You should see all 17 folders listed. If any are missing, re-run the clone command for that node from Step 4.

---

## Step 7 — Restart ComfyUI

ComfyUI will not load the new nodes until it is restarted. In the SimplePod interface, restart your pod or kill and relaunch the ComfyUI process.

On first startup after installing, ComfyUI Manager will run its own dependency check across all nodes. This is normal and will take a minute or two. Let it finish before loading your workflow.

---

## Node Reference

| Node Pack | Provides |
|---|---|
| ComfyUI-VideoHelperSuite | Video loading and saving utilities |
| ComfyUI-Crystools | Switch nodes and resource monitoring |
| ComfyUI-KJNodes | Image resize, utility nodes |
| ComfyUI-MediaMixer | Image batch utilities |
| ComfyUI_tinyterranodes | Float, seed, and string primitives |
| comfyui_controlnet_aux | ControlNet preprocessors (depth, pose, etc.) |
| ComfyUI-Impact-Pack | Masking, segmentation, string utilities |
| ComfyUI-QwenVL | Qwen vision-language model nodes |
| ComfyUI-Easy-Use | Workflow quality-of-life nodes |
| ComfyUI-WanVideoWrapper | WAN video generation nodes |
| ComfyUI-GGUF | GGUF quantized model loader |
| rgthree-comfy | Power Lora Loader, bookmarks, image comparer |
| ComfyUI-Custom-Scripts | Math expressions, show text, and more |
| SeedVR2-VideoUpscaler | SeedVR2 video upscaling nodes |
| ComfyUI-PainterI2V | Image-to-video painter node |
| ComfyUI-Frame-Interpolation | RIFE frame interpolation |
| Bond-Node-Suite | Batch, metadata, utility, and text nodes |

---

## Troubleshooting

**A node shows as missing after restart** — check that the folder exists in `/app/ComfyUI/custom_nodes/` and contains an `__init__.py` file. If the folder is empty or missing, re-clone it.

**ComfyUI Manager keeps reinstalling dependencies on every startup** — this is normal behavior. It checks requirements on each launch but skips anything already installed.

**Out of memory errors** — add `--disable-smart-memory --disable-cuda-malloc` to Docker entrypoint arguments in your SimplePod pod settings.

**SeedVR2 fails to load on non-Blackwell GPUs** — SeedVR2 requires an RTX 5000 series GPU (sm_120). It will fail to import on older hardware. This is expected.

---

*For questions or issues, visit the [Bond Node Suite repository](https://github.com/bondngn-studios/Bond-Node-Suite).*
