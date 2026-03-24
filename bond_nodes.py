import os
import json
import threading
import subprocess
import shutil
import datetime
import re
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths

# --- Shared Utilities & State ---
_STATE = {}
_STATE_LOCK = threading.Lock()

def _abs(p): return os.path.abspath(os.path.expanduser(p))

def _pil_to_tensor_rgb(img):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def _load_json_array(path):
    ap = _abs(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"File not found: {ap}")
    try:
        with open(ap, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        with open(ap, "r") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of strings.")
    return ap, data

# --- CATEGORY CONSTANTS ---
CAT_BATCH    = "Bond/Batch"
CAT_UTIL     = "Bond/Utilities"
CAT_METADATA = "Bond/Metadata"


# ===========================================================================
# BATCH NODES
# ===========================================================================

class PromptJSONSelector:
    """
    Loads a list of prompts from a JSON array or plain text file and returns
    one prompt at a time by index. Designed to work with the Cartesian Index
    Driver idxP output to cycle through prompts in a multi-dimensional batch.

    Supports JSON arrays (["prompt1", "prompt2"]) and TXT files (one prompt
    per line). Use wrap to loop back to the start, or clamp to hold on the
    last prompt when the index exceeds the list size.
    """
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "tooltip": "Absolute path to a .json or .txt file containing your prompts."}),
                "index":     ("INT",    {"default": 0, "min": 0, "tooltip": "Which prompt to return. Wire the idxP output from a Cartesian Index Driver here for batch use."}),
                "mode":      (["auto", "json_array", "txt_lines"], {"tooltip": "auto: detects format automatically. json_array: expects a JSON array of strings. txt_lines: one prompt per line."}),
                "delimiter": ("STRING", {"default": "\n", "tooltip": "Delimiter used to split lines in txt_lines mode. Leave as \\n for standard line breaks."}),
                "wrap_index":(["wrap", "clamp"], {"tooltip": "wrap: loops back to index 0 when the index exceeds the list. clamp: holds on the last prompt."}),
            }
        }

    RETURN_TYPES    = ("STRING", "INT")
    RETURN_NAMES    = ("prompt_text", "count")
    OUTPUT_TOOLTIPS = ("The prompt string at the requested index.", "Total number of prompts in the loaded file.")
    FUNCTION        = "run"
    CATEGORY        = CAT_BATCH

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    @staticmethod
    def _load_list(path, mode, delimiter):
        if not os.path.isfile(path): return []
        mtime = os.path.getmtime(path)
        cache = PromptJSONSelector._cache.get(path)
        if cache and cache[0] == mtime: return cache[1]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        lst = []
        if mode == "auto":
            try:
                data = json.loads(content)
                lst = [str(x) for x in data] if isinstance(data, list) else [l.strip() for l in content.splitlines() if l.strip()]
            except: lst = [l.strip() for l in content.splitlines() if l.strip()]
        elif mode == "json_array":
            try:
                data = json.loads(content)
                if isinstance(data, list): lst = [str(x) for x in data]
            except: lst = []
        else:
            parts = content.split(delimiter) if delimiter and delimiter != "\n" else content.splitlines()
            lst = [p.strip() for p in parts if p.strip()]
        PromptJSONSelector._cache[path] = (mtime, lst)
        return lst

    def run(self, file_path, index, mode, delimiter, wrap_index):
        prompts = self._load_list(file_path, mode, delimiter)
        n = len(prompts)
        if n == 0: return ("", 0)
        i = index % n if wrap_index == "wrap" else max(0, min(index, n - 1))
        return (prompts[i], n)


class BatchIntPick:
    """Picks a single integer value from a batch by index."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("BATCH_INT", {"tooltip": "A batch of integer values to pick from."}),
                "index":  ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": "Zero-based index of the value to return."}),
            },
            "optional": {"wrap": ("BOOLEAN", {"default": True, "tooltip": "If true, wraps around when the index exceeds the batch size. If false, clamps to the last value."})}
        }
    RETURN_TYPES    = ("INT",)
    RETURN_NAMES    = ("value",)
    OUTPUT_TOOLTIPS = ("The integer value at the requested index.",)
    FUNCTION        = "go"
    CATEGORY        = CAT_BATCH

    def go(self, values, index, wrap=True):
        n = len(values) if values is not None else 0
        if n == 0: return (0,)
        i = (index % n) if wrap else max(0, min(index, n - 1))
        try: return (int(values[i]),)
        except: return (0,)


class BatchStringPick:
    """Picks a single string value from a batch by index."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING", {"forceInput": True, "tooltip": "A batch of string values to pick from. Must be wired in — not a text widget."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": "Zero-based index of the string to return."}),
            },
            "optional": {"wrap": ("BOOLEAN", {"default": True, "tooltip": "If true, wraps around when the index exceeds the batch size. If false, clamps to the last value."})}
        }
    RETURN_TYPES    = ("STRING",)
    RETURN_NAMES    = ("text",)
    OUTPUT_TOOLTIPS = ("The string value at the requested index.",)
    FUNCTION        = "go"
    CATEGORY        = CAT_BATCH
    INPUT_IS_LIST   = True

    def go(self, texts, index, wrap=True):
        idx     = index[0] if isinstance(index, list) else index
        do_wrap = wrap[0]  if isinstance(wrap,  list) else wrap
        if not texts: return ("",)
        if isinstance(texts[0], list): texts = texts[0]
        n = len(texts)
        i = (idx % n) if do_wrap else max(0, min(idx, n - 1))
        return (str(texts[i] if texts[i] is not None else ""),)


class BondPromptArrayIterator:
    """
    Iterates through a JSON array of prompt strings one entry at a time,
    advancing the index on each execution. Thread-safe with built-in seed
    syncing so each prompt gets a unique, reproducible seed.

    The index persists between runs until reset is triggered. Use loop to
    cycle back to the start, or leave it off to stop at the last prompt.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_path":            ("STRING",  {"default": "path_to_your_file.json", "tooltip": "Absolute path to a JSON file containing an array of prompt strings."}),
                "base_seed":            ("INT",     {"default": 123456, "min": 0, "max": 0x7FFFFFFF, "tooltip": "Starting seed. Each prompt gets base_seed + index as its seed."}),
                "advance_on_execution": ("BOOLEAN", {"default": True,  "tooltip": "If true, advances to the next prompt after each run."}),
                "pre_advance":          ("BOOLEAN", {"default": True,  "tooltip": "If true, advances the index before returning."}),
                "loop":                 ("BOOLEAN", {"default": False, "tooltip": "If true, wraps back to index 0 after the last prompt."}),
                "reset":                ("BOOLEAN", {"default": False, "tooltip": "Set to true to reset the iterator back to index 0 on the next run."}),
            }
        }
    RETURN_TYPES    = ("STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES    = ("text", "index", "seed", "done")
    OUTPUT_TOOLTIPS = (
        "The prompt string at the current index.",
        "The current zero-based index into the prompt array.",
        "Seed for this prompt: base_seed + index. Wire into your KSampler seed.",
        "True when the iterator has reached the last prompt in the array.",
    )
    FUNCTION = "run"
    CATEGORY = CAT_BATCH

    @classmethod
    def IS_CHANGED(cls, json_path, **kwargs):
        ap  = _abs(json_path)
        st  = _STATE.get(ap)
        idx = st["idx"] if st else -1
        return f"{ap}|idx={idx}|{json.dumps(kwargs)}"

    def run(self, json_path, base_seed=123456, advance_on_execution=True, pre_advance=True, loop=False, reset=False):
        ap, data = _load_json_array(json_path)
        with _STATE_LOCK:
            st = _STATE.get(ap)
            if st is None or reset: _STATE[ap] = {"idx": 0, "size": len(data), "data": data}
            st   = _STATE[ap]
            size = st["size"]
            if size == 0: return ("", 0, base_seed, True)
            idx_to_emit = st["idx"]
            if advance_on_execution and pre_advance:
                st["idx"] = (idx_to_emit + 1) % size if loop else min(idx_to_emit + 1, size - 1)
            prompt = st["data"][idx_to_emit]
            done   = (idx_to_emit == size - 1) and (not loop)
            seed   = int(base_seed) + int(idx_to_emit)
            if advance_on_execution and not pre_advance:
                st["idx"] = (idx_to_emit + 1) % size if loop else min(idx_to_emit + 1, size - 1)
            return (prompt, idx_to_emit, seed, done)


class CartesianIndexDriverImg:
    """
    Maps a single master counter i to a 3-dimensional grid of indices.
    Use this to drive batch workflows where you want to run every combination
    of N1 x N2 x N3 items — for example, every model against every background
    against every lighting setup.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "i":  ("INT", {"default": 0, "tooltip": "Master counter. Typically wired from a queue index or Range Stepper."}),
            "N1": ("INT", {"default": 1, "tooltip": "Size of dimension 1 (slowest changing)."}),
            "N2": ("INT", {"default": 1, "tooltip": "Size of dimension 2 (middle)."}),
            "N3": ("INT", {"default": 1, "tooltip": "Size of dimension 3 (fastest changing)."}),
        }}
    RETURN_TYPES    = ("INT", "INT", "INT", "INT")
    RETURN_NAMES    = ("idx1", "idx2", "idx3", "total")
    OUTPUT_TOOLTIPS = (
        "Index into dimension 1 (slowest changing).",
        "Index into dimension 2 (middle).",
        "Index into dimension 3 (fastest changing).",
        "Total number of combinations: N1 x N2 x N3. Use as your queue count.",
    )
    FUNCTION = "run"
    CATEGORY = CAT_BATCH

    def run(self, i, N1, N2, N3):
        total = N1 * N2 * N3
        i = min(i, total - 1) if total > 0 else 0
        return (i // (N2 * N3), (i // N3) % N2, i % N3, total)


class CartesianIndexDriverImgPrmpt:
    """
    Maps a single master counter i to a 4-dimensional grid covering 3 image
    dimensions plus a prompt dimension. Use prompt_fastest to cycle through all
    prompts for each image combination, or prompt_slowest for the reverse.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "i":            ("INT", {"default": 0, "tooltip": "Master counter."}),
            "N1":           ("INT", {"default": 1, "tooltip": "Size of image dimension 1."}),
            "N2":           ("INT", {"default": 1, "tooltip": "Size of image dimension 2."}),
            "N3":           ("INT", {"default": 1, "tooltip": "Size of image dimension 3."}),
            "Np":           ("INT", {"default": 1, "tooltip": "Number of prompts. Wire idxP into a Prompt JSON/TXT Selector."}),
            "order":        (["prompt_fastest", "prompt_slowest"], {"tooltip": "prompt_fastest: cycles all prompts before next image combination. prompt_slowest: cycles all image combinations before next prompt."}),
            "overflow_mode":(["clamp", "wrap"], {"tooltip": "clamp: hold on last combination when i exceeds total. wrap: loop back to start."}),
        }}
    RETURN_TYPES    = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES    = ("idx1", "idx2", "idx3", "idxP", "total")
    OUTPUT_TOOLTIPS = (
        "Index into image dimension 1.",
        "Index into image dimension 2.",
        "Index into image dimension 3.",
        "Index into the prompt dimension. Wire into a Prompt JSON/TXT Selector.",
        "Total combinations: N1 x N2 x N3 x Np. Use as your queue count.",
    )
    FUNCTION = "run"
    CATEGORY = CAT_BATCH

    def run(self, i, N1, N2, N3, Np, order, overflow_mode):
        total = N1 * N2 * N3 * Np
        if total <= 0: return (0, 0, 0, 0, 0)
        i_eff = i % total if overflow_mode == "wrap" else max(0, min(i, total - 1))
        if order == "prompt_slowest": sS, sO, sB, sP = 1, N1, N1 * N2, N1 * N2 * N3
        else:                         sP, sB, sO, sS = 1, Np, N3 * Np, N2 * N3 * Np
        return ((i_eff // sS) % N1, (i_eff // sO) % N2, (i_eff // sB) % N3, (i_eff // sP) % Np, total)


class RangeStepper:
    """
    A stateful counter that auto-increments by step on every queue run.
    State persists between runs. Use reset to return to the start value.
    """
    _state = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT",     {"default": 0,     "tooltip": "The value to start from, and the value returned after a reset."}),
                "stop":  ("INT",     {"default": 0,     "tooltip": "The maximum value. The counter will not exceed this."}),
                "step":  ("INT",     {"default": 1,     "tooltip": "How much to increment by on each run."}),
                "reset": ("BOOLEAN", {"default": False, "tooltip": "Set to true to reset the counter back to start on the next run."}),
            },
            "optional": {"trigger": ("*", {"tooltip": "Wire any output here to force this node to run after it."})}
        }
    RETURN_TYPES    = ("INT",)
    RETURN_NAMES    = ("i",)
    OUTPUT_TOOLTIPS = ("The current counter value. Increments by step on each run until it reaches stop.",)
    FUNCTION        = "run"
    CATEGORY        = CAT_BATCH

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def run(self, start, stop, step, reset=False, trigger=None):
        key   = (start, stop, step)
        cur   = start if reset or key not in self._state else self._state[key]
        out_i = max(start, min(cur, stop))
        self._state[key] = min(out_i + step, stop)
        return (out_i,)


# ===========================================================================
# UTILITY NODES
# ===========================================================================

class BondSwitch2to1:
    """
    Routes one of two inputs to a single output based on a toggle.
    Works with any data type — images, strings, latents, masks, anything.

    When select_a_or_b is false (A), input_a passes through.
    When select_a_or_b is true  (B), input_b passes through.

    Use case: toggle between two prompts, two models, two branches, or
    any two values in your workflow without rewiring.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_a": ("*", {"tooltip": "The value to pass through when output is set to A."}),
                "input_b": ("*", {"tooltip": "The value to pass through when output is set to B."}),
                "output":  (["A", "B"], {"default": "A", "tooltip": "A = pass input_a through. B = pass input_b through."}),
            }
        }
    RETURN_TYPES    = ("*",)
    RETURN_NAMES    = ("output",)
    OUTPUT_TOOLTIPS = ("The selected input — either input_a or input_b depending on the selection.",)
    FUNCTION        = "switch"
    CATEGORY        = CAT_UTIL

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def switch(self, input_a, input_b, output):
        return (input_b if output == "B" else input_a,)


class BondSwitch1to2:
    """
    Routes a single input to one of two outputs based on a dropdown selection.
    Works with any data type — images, strings, latents, masks, anything.

    When output is A, the input is sent to output_a (output_b is None).
    When output is B, the input is sent to output_b (output_a is None).

    Use case: send your image to either a regular Save Image node or a
    Bond: Save With Custom Metadata node by changing the dropdown — no
    rewiring required.

    Note: the inactive output will be None. Output nodes like Save Image
    will gracefully skip execution when their inputs are None.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input":  ("*",        {"tooltip": "The value to route. Goes to output_a when A, output_b when B."}),
                "output": (["A", "B"], {"default": "A", "tooltip": "A = route to output_a. B = route to output_b."}),
            }
        }

    RETURN_TYPES    = ("*",  "*",)
    RETURN_NAMES    = ("output_a", "output_b",)
    OUTPUT_TOOLTIPS = (
        "Active when output is set to A. None when B is selected.",
        "Active when output is set to B. None when A is selected.",
    )
    FUNCTION        = "switch"
    CATEGORY        = CAT_UTIL

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def switch(self, input, output):
        return (None, input) if output == "B" else (input, None)


class BondLoadImage:
    """
    Upload-based image loader with a 'choose file to upload' button and image
    preview, identical to the native ComfyUI Load Image node. Adds path, stem,
    and dir outputs so it wires directly into Bond metadata nodes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted([
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith(".")
        ]) if os.path.exists(input_dir) else []
        return {"required": {"image": (sorted(files), {"image_upload": True, "tooltip": "Click 'choose file to upload' to select an image."})}}

    RETURN_TYPES    = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES    = ("image", "path", "stem", "dir")
    OUTPUT_TOOLTIPS = (
        "The loaded image as a tensor.",
        "Full absolute path to the uploaded file on disk.",
        "Filename without extension. e.g. 'my_photo' from 'my_photo.jpg'.",
        "Directory containing the file.",
    )
    FUNCTION = "load"
    CATEGORY = CAT_UTIL

    @classmethod
    def IS_CHANGED(cls, image):
        fp = os.path.join(folder_paths.get_input_directory(), image)
        if os.path.isfile(fp):
            import hashlib
            with open(fp, "rb") as f: return hashlib.md5(f.read()).hexdigest()
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        fp = os.path.join(folder_paths.get_input_directory(), image)
        return True if os.path.isfile(fp) else f"File not found: {image}"

    def load(self, image):
        fp  = os.path.join(folder_paths.get_input_directory(), image)
        img = ImageOps.exif_transpose(Image.open(fp))
        return (_pil_to_tensor_rgb(img), fp, os.path.splitext(os.path.basename(fp))[0], os.path.dirname(fp))


class LoadImageFromPath:
    """
    Loads an image from a typed absolute file path. Best for batch workflows
    where files are already on disk. Outputs the same path/stem/dir values as
    Bond: Load Image so both nodes wire identically into metadata nodes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "", "tooltip": "Full absolute path to the image file. e.g. C:\\Users\\you\\Pictures\\photo.jpg"})}}

    RETURN_TYPES    = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES    = ("image", "path", "stem", "dir")
    OUTPUT_TOOLTIPS = (
        "The loaded image as a tensor.",
        "Full absolute path to the file as provided.",
        "Filename without extension.",
        "Directory containing the file.",
    )
    FUNCTION = "load"
    CATEGORY = CAT_UTIL

    def load(self, file_path):
        ap = _abs(file_path.strip())
        if not os.path.isfile(ap): raise FileNotFoundError(f"Not found: {ap}")
        return (_pil_to_tensor_rgb(Image.open(ap)), ap, os.path.splitext(os.path.basename(ap))[0], os.path.dirname(ap))


class LoadVideoFromPath:
    """
    Passes a video filepath into the workflow as a string. Validates the file
    exists and provides path, stem, and directory as outputs. Wire path into a
    video-aware node or Bond: Save Video With Metadata.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "", "tooltip": "Full absolute path to the video file. e.g. C:\\Users\\you\\Videos\\clip.mp4"})}}

    RETURN_TYPES    = ("STRING", "STRING", "STRING")
    RETURN_NAMES    = ("path", "stem", "dir")
    OUTPUT_TOOLTIPS = (
        "Full absolute path to the video file.",
        "Filename without extension.",
        "Directory containing the file.",
    )
    FUNCTION = "load"
    CATEGORY = CAT_UTIL

    def load(self, file_path):
        ap = _abs(file_path.strip())
        if not os.path.isfile(ap): raise FileNotFoundError(f"Not found: {ap}")
        return (ap, os.path.splitext(os.path.basename(ap))[0], os.path.dirname(ap))


class BondBatchImageLoader:
    """
    Loads one image per execution from a directory. Three modes:
    single_image (by index), sequential (auto-advances each run), random (by seed).
    Sequential state persists between runs.
    """
    SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode":                   (["single_image", "sequential", "random"], {"tooltip": "single_image: by index. sequential: auto-advances. random: by seed."}),
            "seed":                   ("INT",     {"default": 0, "min": 0, "max": 0x7FFFFFFF, "tooltip": "Seed used only in random mode."}),
            "index":                  ("INT",     {"default": 0, "min": 0, "max": 100_000, "tooltip": "Used only in single_image mode. Zero-based."}),
            "path":                   ("STRING",  {"default": "", "tooltip": "Absolute path to the directory containing your images."}),
            "pattern":                ("STRING",  {"default": "*", "tooltip": "Filename filter. Use * for all, or e.g. 'photo_*' for a prefix."}),
            "allow_RGBA_output":      ("BOOLEAN", {"default": False, "tooltip": "If true, loads as RGBA preserving transparency. If false, converts to RGB."}),
            "filename_text_extension":("BOOLEAN", {"default": True,  "tooltip": "If true, filename_text includes the extension. If false, extension is stripped."}),
        }}

    RETURN_TYPES    = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES    = ("image", "filename_text", "index", "total")
    OUTPUT_TOOLTIPS = (
        "The loaded image as a tensor.",
        "The filename of the loaded image.",
        "The index of the image that was loaded this run.",
        "Total number of images in the directory matching the pattern.",
    )
    FUNCTION = "load"
    CATEGORY = CAT_UTIL

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def _scan_dir(self, path, pattern):
        import fnmatch
        ap = _abs(path)
        if not os.path.isdir(ap): raise ValueError(f"Directory not found: {ap}")
        files = sorted([
            os.path.join(ap, f) for f in os.listdir(ap)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXT
            and fnmatch.fnmatch(f, pattern if pattern.strip() else "*")
        ])
        if not files: raise ValueError(f"No supported images found in: {ap}")
        return files

    def _open(self, path, allow_rgba):
        img = Image.open(path)
        arr = np.array(img.convert("RGBA" if allow_rgba else "RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def load(self, mode, seed, index, path, pattern, allow_RGBA_output, filename_text_extension, control_after_generate="fixed"):
        files = self._scan_dir(path, pattern)
        n = len(files)
        if mode == "single_image":
            i = index % n
        elif mode == "sequential":
            key = _abs(path)
            with _STATE_LOCK:
                st = _STATE.setdefault(f"batchloader:{key}", {"idx": 0})
                i  = st["idx"]
                st["idx"] = (i + 1) % n
        else:
            i = int(np.random.default_rng(seed).integers(0, n))
        tensor = self._open(files[i], allow_RGBA_output)
        fname  = os.path.basename(files[i]) if filename_text_extension else os.path.splitext(os.path.basename(files[i]))[0]
        return (tensor, fname, i, n)


class BondShowText:
    """
    Displays a string input as a text area rendered directly inside the node
    body. Requires the Bond JS extension (js/bond_nodes.js) to render inline.
    Text persists across page reloads. Passes the string through as text_out.
    """
    CATEGORY     = CAT_UTIL
    FUNCTION     = "show_text"
    OUTPUT_NODE  = True
    RETURN_TYPES    = ("STRING",)
    RETURN_NAMES    = ("text_out",)
    OUTPUT_TOOLTIPS = ("The input string passed through unchanged.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_in": ("STRING", {"forceInput": True, "tooltip": "Any string output. The text will be displayed inside this node's body."})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def show_text(self, text_in=""):
        return {"ui": {"text": [text_in]}, "result": (text_in,)}


# ===========================================================================
# SHARED METADATA HELPERS
# ===========================================================================

CAMERA_PRESETS = {
    "Manual": {},

    "iPhone 15 Pro": {
        # Device
        "Make": "Apple", "Model": "iPhone 15 Pro", "HostComputer": "iPhone 15 Pro",
        # Lens / optics
        "LensModel": "iPhone 15 Pro back triple camera 6.765mm f/1.78", "LensMake": "Apple",
        "FNumber": "1.78", "ApertureValue": "1.78", "FocalLength": "6.765 mm", "FocalLengthIn35mmFormat": "24 mm",
        # Exposure
        "ExposureProgram": "Normal program", "ExposureMode": "Auto", "MeteringMode": "Multi-segment",
        "ShutterSpeedValue": "1/120", "BrightnessValue": "4.7", "ExposureCompensation": "0", "ISOSpeedRatings": "50",
        # White balance / color
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        # Flash
        "Flash": "No flash function",
        # Software / processing
        "Software": "17.0", "XMP:CreatorTool": "17.0", "ProcessingSoftware": "17.0",
        # Orientation & resolution
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "72", "YResolution": "72",
        # Subsecond & timezone
        "SubSecTimeOriginal": "042", "SubSecTimeDigitized": "042", "OffsetTimeOriginal": "-05:00", "OffsetTimeDigitized": "-05:00",
        # Sensing
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Custom process",
    },

    "iPhone 14 Pro": {
        "Make": "Apple", "Model": "iPhone 14 Pro", "HostComputer": "iPhone 14 Pro",
        "LensModel": "iPhone 14 Pro back triple camera 6.86mm f/1.78", "LensMake": "Apple",
        "FNumber": "1.78", "ApertureValue": "1.78", "FocalLength": "6.86 mm", "FocalLengthIn35mmFormat": "24 mm",
        "ExposureProgram": "Normal program", "ExposureMode": "Auto", "MeteringMode": "Multi-segment",
        "ShutterSpeedValue": "1/120", "BrightnessValue": "4.5", "ExposureCompensation": "0", "ISOSpeedRatings": "50",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "No flash function",
        "Software": "16.0", "XMP:CreatorTool": "16.0", "ProcessingSoftware": "16.0",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "72", "YResolution": "72",
        "SubSecTimeOriginal": "031", "SubSecTimeDigitized": "031", "OffsetTimeOriginal": "-05:00", "OffsetTimeDigitized": "-05:00",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Custom process",
    },

    "iPhone 13": {
        "Make": "Apple", "Model": "iPhone 13", "HostComputer": "iPhone 13",
        "LensModel": "iPhone 13 back dual wide camera 5.1mm f/1.6", "LensMake": "Apple",
        "FNumber": "1.6", "ApertureValue": "1.6", "FocalLength": "5.1 mm", "FocalLengthIn35mmFormat": "26 mm",
        "ExposureProgram": "Normal program", "ExposureMode": "Auto", "MeteringMode": "Multi-segment",
        "ShutterSpeedValue": "1/100", "BrightnessValue": "4.2", "ExposureCompensation": "0", "ISOSpeedRatings": "64",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "No flash function",
        "Software": "15.0", "XMP:CreatorTool": "15.0", "ProcessingSoftware": "15.0",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "72", "YResolution": "72",
        "SubSecTimeOriginal": "018", "SubSecTimeDigitized": "018", "OffsetTimeOriginal": "-05:00", "OffsetTimeDigitized": "-05:00",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Custom process",
    },

    "Samsung Galaxy S24 Ultra": {
        "Make": "Samsung", "Model": "SM-S928B", "HostComputer": "SM-S928B",
        "LensModel": "Samsung Galaxy S24 Ultra rear camera 6.3mm f/1.7", "LensMake": "Samsung",
        "FNumber": "1.7", "ApertureValue": "1.7", "FocalLength": "6.3 mm", "FocalLengthIn35mmFormat": "23 mm",
        "ExposureProgram": "Normal program", "ExposureMode": "Auto", "MeteringMode": "Center-weighted average",
        "ShutterSpeedValue": "1/100", "BrightnessValue": "4.0", "ExposureCompensation": "0", "ISOSpeedRatings": "50",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "No flash function",
        "Software": "Android 14", "XMP:CreatorTool": "Android 14",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "72", "YResolution": "72",
        "SubSecTimeOriginal": "027", "SubSecTimeDigitized": "027",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Normal",
    },

    "Google Pixel 8 Pro": {
        "Make": "Google", "Model": "Pixel 8 Pro", "HostComputer": "Pixel 8 Pro",
        "LensModel": "Google Pixel 8 Pro rear camera 6.81mm f/1.68", "LensMake": "Google",
        "FNumber": "1.68", "ApertureValue": "1.68", "FocalLength": "6.81 mm", "FocalLengthIn35mmFormat": "24 mm",
        "ExposureProgram": "Normal program", "ExposureMode": "Auto", "MeteringMode": "Center-weighted average",
        "ShutterSpeedValue": "1/120", "BrightnessValue": "4.3", "ExposureCompensation": "0", "ISOSpeedRatings": "50",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "No flash function",
        "Software": "Android 14", "XMP:CreatorTool": "Android 14",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "72", "YResolution": "72",
        "SubSecTimeOriginal": "009", "SubSecTimeDigitized": "009",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Normal",
    },

    "Sony A7R V": {
        "Make": "Sony", "Model": "ILCE-7RM5", "HostComputer": "",
        "LensModel": "FE 50mm F1.4 GM", "LensMake": "Sony",
        "FNumber": "2.8", "ApertureValue": "2.8", "FocalLength": "50 mm", "FocalLengthIn35mmFormat": "50 mm",
        "ExposureProgram": "Aperture-priority AE", "ExposureMode": "Auto", "MeteringMode": "Multi-segment",
        "ShutterSpeedValue": "1/250", "BrightnessValue": "5.1", "ExposureCompensation": "0", "ISOSpeedRatings": "100",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "Off, Did not fire",
        "Software": "ILCE-7RM5 v2.00", "XMP:CreatorTool": "ILCE-7RM5 v2.00",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "350", "YResolution": "350",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Normal",
    },

    "Canon EOS R5": {
        "Make": "Canon", "Model": "Canon EOS R5", "HostComputer": "",
        "LensModel": "RF50mm F1.2 L USM", "LensMake": "Canon",
        "FNumber": "2.8", "ApertureValue": "2.8", "FocalLength": "50 mm", "FocalLengthIn35mmFormat": "50 mm",
        "ExposureProgram": "Aperture-priority AE", "ExposureMode": "Auto", "MeteringMode": "Evaluative",
        "ShutterSpeedValue": "1/250", "BrightnessValue": "5.0", "ExposureCompensation": "0", "ISOSpeedRatings": "100",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "Off, Did not fire",
        "Software": "Digital Photo Professional 4.20.10", "XMP:CreatorTool": "Digital Photo Professional 4.20.10",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "72", "YResolution": "72",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Normal",
    },

    "Nikon Z9": {
        "Make": "Nikon", "Model": "NIKON Z 9", "HostComputer": "",
        "LensModel": "NIKKOR Z 50mm f/1.2 S", "LensMake": "Nikon",
        "FNumber": "2.8", "ApertureValue": "2.8", "FocalLength": "50 mm", "FocalLengthIn35mmFormat": "50 mm",
        "ExposureProgram": "Aperture-priority AE", "ExposureMode": "Auto", "MeteringMode": "Multi-segment",
        "ShutterSpeedValue": "1/250", "BrightnessValue": "5.2", "ExposureCompensation": "0", "ISOSpeedRatings": "100",
        "WhiteBalance": "Auto", "ColorSpace": "sRGB", "SceneCaptureType": "Standard", "SceneType": "Directly photographed",
        "Flash": "Off, Did not fire",
        "Software": "Ver.3.10", "XMP:CreatorTool": "Ver.3.10",
        "Orientation": "Horizontal (normal)", "ResolutionUnit": "inches", "XResolution": "300", "YResolution": "300",
        "SensingMethod": "One-chip color area sensor", "CustomRendered": "Normal",
    },
}

PRESET_NAMES = list(CAMERA_PRESETS.keys())

_SUMMARY_SKIP = {
    "SourceFile", "ExifToolVersion", "FileSize", "FileModifyDate", "FileAccessDate",
    "FileInodeChangeDate", "FilePermissions", "FileType", "FileTypeExtension",
    "MIMEType", "Directory", "ThumbnailImage", "ThumbnailLength", "ThumbnailOffset",
}

_COMMON_TAGS = {
    "Make", "Model", "LensModel", "LensMake", "FNumber", "ApertureValue",
    "FocalLength", "FocalLengthIn35mmFormat", "ExposureProgram", "ExposureMode",
    "MeteringMode", "ShutterSpeedValue", "BrightnessValue", "ExposureCompensation",
    "ISOSpeedRatings", "WhiteBalance", "ColorSpace", "SceneCaptureType", "SceneType",
    "Flash", "Software", "DateTimeOriginal", "CreateDate", "ModifyDate",
    "GPSLatitude", "GPSLongitude", "GPSLatitudeRef", "GPSLongitudeRef",
    "ImageDescription", "Artist", "Copyright", "Keywords", "Description",
    "Creator", "Rights", "Subject", "Author", "HostComputer",
    "Orientation", "ResolutionUnit", "XResolution", "YResolution",
    "SensingMethod", "CustomRendered", "ImageWidth", "ImageHeight", "FileSize", "FileType",
}

_WORKFLOW_NOISE = {"Prompt", "Workflow", "workflow", "prompt", "PNG:Prompt", "PNG:Workflow"}


def _resolve_exiftool(exiftool_path: str) -> str:
    candidate = exiftool_path.strip()
    if os.path.isfile(candidate): return candidate
    found = shutil.which(candidate)
    if found: return found
    raise RuntimeError(
        f"[BondNodes] Cannot find exiftool at '{candidate}'.\n"
        "  Windows: C:\\exiftool\\exiftool.exe\n"
        "  Mac/Linux: /usr/local/bin/exiftool\n"
        "Download: https://exiftool.org/"
    )

def _geocode_city(city_name: str):
    try:
        import urllib.request, urllib.parse
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": city_name, "format": "json", "limit": "1"})
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-BondNodes/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if data: return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception: pass
    return None, None

def _decimal_to_dms(deg: float) -> str:
    d = int(abs(deg)); m_float = (abs(deg) - d) * 60; m = int(m_float); s = (m_float - m) * 60
    return f"{d} {m} {s:.6f}"

def _gps_args(city_name: str) -> list:
    if not city_name.strip(): return []
    lat, lon = _geocode_city(city_name.strip())
    if lat is None:
        print(f"[BondNodes] Warning: could not geocode '{city_name}' — GPS skipped.")
        return []
    return [
        f"-GPSLatitude={_decimal_to_dms(lat)}", f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-GPSLongitude={_decimal_to_dms(lon)}", f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
    ]

def _read_metadata_summary(exiftool_exe: str, filepath: str, filter_common: bool = True) -> str:
    result = subprocess.run([exiftool_exe, "-json", "-a", "-G1", filepath], capture_output=True, text=True)
    if result.returncode != 0: return f"(exiftool error: {result.stderr.strip()})"
    try:
        meta = json.loads(result.stdout.strip())
        meta = meta[0] if meta else {}
    except Exception: return "(could not parse exiftool output)"
    lines = []
    for key, value in meta.items():
        bare = key.split(":")[-1] if ":" in key else key
        if bare in _SUMMARY_SKIP: continue
        if filter_common and bare not in _COMMON_TAGS: continue
        sv = str(value)
        if len(sv) > 100: sv = sv[:97] + "..."
        lines.append(f"{key:<36} {sv}")
    return "\n".join(lines) if lines else "(no metadata found)"

def _read_metadata_json(exiftool_exe: str, filepath: str, strip_workflow: bool = False) -> str:
    result = subprocess.run([exiftool_exe, "-json", "-a", "-G1", filepath], capture_output=True, text=True)
    if result.returncode != 0: return "{}"
    try:
        parsed = json.loads(result.stdout.strip())
        meta   = parsed[0] if parsed else {}
    except Exception: return "{}"
    if strip_workflow:
        for k in [k for k in meta if any(n in k for n in _WORKFLOW_NOISE)]:
            meta.pop(k, None)
    return json.dumps(meta, indent=2, ensure_ascii=False)

def _common_exiftool_args(exiftool_exe, filepath, preset_tags,
                          manual_make="", manual_model="", manual_lens="",
                          city_name="", datetime_str="", description="", artist="",
                          copyright_str="", keywords="", custom_xmp="", is_video=False) -> list:
    args = [exiftool_exe, "-overwrite_original"]
    tags = dict(preset_tags)
    if manual_make:  tags["Make"]      = manual_make
    if manual_model: tags["Model"]     = manual_model
    if manual_lens:  tags["LensModel"] = manual_lens
    for tag, value in tags.items():
        if value:
            args.append(f"-{tag}={value}")
            if is_video: args.append(f"-QuickTime:{tag}={value}")
    args += _gps_args(city_name)
    dt = datetime_str.strip() or datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    args += [f"-DateTimeOriginal={dt}", f"-CreateDate={dt}", f"-ModifyDate={dt}"]
    if is_video: args += [f"-QuickTime:CreateDate={dt}", f"-QuickTime:ModifyDate={dt}"]
    if description.strip():
        args += [f"-ImageDescription={description}", f"-XMP:Description={description}"]
        args.append(f"-IPTC:Caption-Abstract={description}" if not is_video else f"-QuickTime:Description={description}")
    if artist.strip():
        args += [f"-Artist={artist}", f"-XMP:Creator={artist}"]
        args.append(f"-IPTC:By-line={artist}" if not is_video else f"-QuickTime:Author={artist}")
    if copyright_str.strip():
        args += [f"-Copyright={copyright_str}", f"-XMP:Rights={copyright_str}"]
        args.append(f"-IPTC:CopyrightNotice={copyright_str}" if not is_video else f"-QuickTime:Copyright={copyright_str}")
    if keywords.strip():
        for kw in re.split(r"[,;]+", keywords):
            kw = kw.strip()
            if kw: args += [f"-Keywords={kw}", f"-XMP:Subject={kw}"]
    if custom_xmp.strip():
        for line in custom_xmp.splitlines():
            line = line.strip()
            if "=" in line:
                t, _, v = line.partition("=")
                args.append(f"-XMP:{t.strip()}={v.strip()}")
    args.append(filepath)
    return args

def _run_exiftool(args: list):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[BondNodes] exiftool error: {result.stderr}")
        return False
    print(f"[BondNodes] exiftool: {result.stdout.strip()}")
    return True

def _split_prefix(filename_prefix: str):
    """Split 'output/bond/img' → (resolved_dir, 'img'). Mirrors native SaveImage."""
    filename_prefix = filename_prefix.strip()
    directory = os.path.dirname(filename_prefix)
    prefix    = os.path.basename(filename_prefix) or "bond"
    if not directory: directory = "output"
    out_dir = directory if os.path.isabs(directory) else os.path.join(folder_paths.get_output_directory(), directory)
    return out_dir, prefix

def _build_written_summary(camera_preset, preset_tags, override_make, override_model,
                            override_lens, location_city, datetime_override, description,
                            artist, copyright, keywords, custom_xmp_tags, saved_paths, is_video=False):
    merged = dict(preset_tags)
    if override_make:  merged["Make"]      = override_make
    if override_model: merged["Model"]     = override_model
    if override_lens:  merged["LensModel"] = override_lens
    dt    = datetime_override.strip() or datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    lines = [f"{'🎬' if is_video else '📷'} Metadata Written", "─" * 36]
    lines.append(f"{'Preset':<16} {camera_preset}")
    for tag, val in merged.items():
        if val: lines.append(f"{tag:<16} {val}")
    lines.append(f"{'DateTime':<16} {dt}")
    if location_city.strip():  lines.append(f"{'Location':<16} {location_city}")
    if description.strip():    lines.append(f"{'Description':<16} {description}")
    if artist.strip():         lines.append(f"{'Artist':<16} {artist}")
    if copyright.strip():      lines.append(f"{'Copyright':<16} {copyright}")
    if keywords.strip():       lines.append(f"{'Keywords':<16} {keywords}")
    if custom_xmp_tags.strip():
        lines.append("Custom XMP:")
        for line in custom_xmp_tags.splitlines():
            if line.strip(): lines.append(f"  {line.strip()}")
    lines.append("─" * 36)
    if len(saved_paths) == 1: lines.append(f"File: {saved_paths[0]}")
    else:
        lines.append("Files:")
        for p in saved_paths: lines.append(f"  {p}")
    return "\n".join(lines)

# Shared INPUT_TYPES block for camera + rights fields (used by both save nodes)
def _camera_and_rights_inputs():
    return {
        "camera_preset":     (PRESET_NAMES, {"default": "iPhone 15 Pro", "tooltip": "Camera preset. Sets a full realistic set of EXIF tags. Choose Manual to use the override fields."}),
        "override_make":     ("STRING", {"default": "", "multiline": False, "tooltip": "Override the Make tag. Leave blank to use the preset."}),
        "override_model":    ("STRING", {"default": "", "multiline": False, "tooltip": "Override the Model tag. Leave blank to use the preset."}),
        "override_lens":     ("STRING", {"default": "", "multiline": False, "tooltip": "Override the LensModel tag. Leave blank to use the preset."}),
        "location_city":     ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. Nashville, TN  or  Eiffel Tower, Paris", "tooltip": "City or landmark geocoded into GPS via Nominatim. Requires internet. Leave blank to skip."}),
        "datetime_override": ("STRING", {"default": "", "multiline": False, "placeholder": "Blank = now  |  YYYY:MM:DD HH:MM:SS", "tooltip": "Date/time to stamp. Blank = current system time."}),
        "description":       ("STRING", {"default": "", "multiline": True,  "tooltip": "Written to EXIF ImageDescription, XMP Description, and IPTC Caption."}),
        "artist":            ("STRING", {"default": "", "multiline": False, "tooltip": "Written to EXIF Artist, XMP Creator, and IPTC By-line."}),
        "copyright":         ("STRING", {"default": "", "multiline": False, "tooltip": "Written to EXIF Copyright, XMP Rights, and IPTC CopyrightNotice. e.g. © 2025 Your Name"}),
        "keywords":          ("STRING", {"default": "", "multiline": False, "placeholder": "comma or semicolon separated", "tooltip": "Written to EXIF Keywords and XMP Subject."}),
        "custom_xmp_tags":   ("STRING", {"default": "", "multiline": True,  "placeholder": "One per line: TagName=Value\ne.g. Rating=5\nLabel=Red", "tooltip": "Additional XMP tags in TagName=Value format, one per line."}),
    }


# ===========================================================================
# METADATA NODES
# ===========================================================================

class BondReadMetadata:
    """
    Reads all metadata from an image or video file using exiftool and returns
    it as both a human-readable summary and raw JSON. Use common_only to see
    the most useful tags, or all to see everything including file system data.

    The filepath passes through unchanged for chaining into Strip Metadata.

    Requires exiftool installed and the path set in the exiftool_path widget.
    """
    CATEGORY     = CAT_METADATA
    FUNCTION     = "read_metadata"
    OUTPUT_NODE  = True
    RETURN_TYPES    = ("STRING", "STRING", "STRING",)
    RETURN_NAMES    = ("filepath", "metadata_summary", "metadata_json",)
    OUTPUT_TOOLTIPS = (
        "The input filepath passed through unchanged. Wire into Strip Metadata or other nodes.",
        "Human-readable formatted metadata. Wire into Bond: Show Text.",
        "Complete raw JSON from exiftool. Contains every tag and value found in the file.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "filepath":       ("STRING", {"default": "", "multiline": False, "tooltip": "Absolute path to the file to read. Wire from a load node's path output."}),
            "exiftool_path":  ("STRING", {"default": "exiftool", "multiline": False, "placeholder": "exiftool  or  C:\\exiftool\\exiftool.exe", "tooltip": "Path to exiftool. Leave as 'exiftool' if on your PATH."}),
            "summary_filter": (["common_only", "all"], {"default": "common_only", "tooltip": "common_only: camera, GPS, date, rights. all: everything exiftool finds."}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def read_metadata(self, filepath, exiftool_path, summary_filter):
        filepath = filepath.strip()
        if not filepath:               return ("", "No filepath provided.", "{}")
        if not os.path.isfile(filepath): return (filepath, f"File not found: {filepath}", "{}")
        exiftool_exe  = _resolve_exiftool(exiftool_path)
        filter_common = (summary_filter == "common_only")
        summary  = _read_metadata_summary(exiftool_exe, filepath, filter_common)
        raw_json = _read_metadata_json(exiftool_exe, filepath, strip_workflow=False)
        lines    = [f"🔍 Metadata: {os.path.basename(filepath)}", "─" * 36, summary, "─" * 36]
        return (filepath, "\n".join(lines), raw_json)


class BondStripMetadata:
    """
    Strips ALL metadata from an image or video file in-place using exiftool.
    Destructive — use backup_original to save a .meta_backup copy first.

    Works with PNG, JPEG, WEBP, MP4, MOV, MKV, and most exiftool formats.
    Filepath passes through for chaining.

    Requires exiftool installed and the path set in the exiftool_path widget.
    """
    CATEGORY     = CAT_METADATA
    FUNCTION     = "strip_metadata"
    OUTPUT_NODE  = True
    RETURN_TYPES    = ("STRING", "STRING",)
    RETURN_NAMES    = ("filepath", "status",)
    OUTPUT_TOOLTIPS = (
        "The input filepath passed through unchanged.",
        "Status message confirming the strip completed, including backup path if enabled.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "filepath":        ("STRING",  {"default": "", "multiline": False, "tooltip": "Absolute path to the file to strip. Wire from a load node's path output."}),
            "exiftool_path":   ("STRING",  {"default": "exiftool", "multiline": False, "placeholder": "exiftool  or  C:\\exiftool\\exiftool.exe", "tooltip": "Path to exiftool. Leave as 'exiftool' if on your PATH."}),
            "backup_original": ("BOOLEAN", {"default": False, "tooltip": "If true, saves filename.ext.meta_backup before stripping."}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def strip_metadata(self, filepath, exiftool_path, backup_original):
        filepath = filepath.strip()
        if not filepath:               return ("", "No filepath provided.")
        if not os.path.isfile(filepath): return (filepath, f"File not found: {filepath}")
        exiftool_exe = _resolve_exiftool(exiftool_path)
        if backup_original:
            shutil.copy2(filepath, filepath + ".meta_backup")
            print(f"[BondStripMetadata] Backup saved: {filepath}.meta_backup")
        result = subprocess.run([exiftool_exe, "-all=", "-overwrite_original", filepath], capture_output=True, text=True)
        if result.returncode != 0: return (filepath, f"exiftool error: {result.stderr}")
        status = (f"✅ Metadata stripped: {os.path.basename(filepath)}\n"
                  f"{'Backup: ' + filepath + '.meta_backup' if backup_original else 'No backup'}")
        print(f"[BondStripMetadata] {result.stdout.strip()}")
        return (filepath, status)


class BondSaveWithCustomMetadata:
    """
    All-in-one metadata replacement node for images. Reads the original
    metadata from the source file, saves a fresh PNG from the image tensor,
    stamps it with your custom metadata, then reads it back to confirm.

    The filename_prefix works like native ComfyUI Save Image:
      output/bond/img  →  saves to output/bond/  named img_[timestamp].png
      E:/photos/shoot  →  saves to E:/photos/    named shoot_[timestamp].png

    Camera presets include realistic EXIF data for iPhone, Samsung, Pixel,
    Sony, Canon, and Nikon. GPS is geocoded from a city name via Nominatim
    (requires internet access).

    Requires exiftool installed and the path set in the exiftool_path widget.

    Intended wiring:
        Bond: Load Image  OR  Bond: Load Image From Path
          image  →  images
          path   →  source_filepath
    """
    CATEGORY     = CAT_METADATA
    FUNCTION     = "save_with_custom_metadata"
    OUTPUT_NODE  = True
    RETURN_TYPES    = ("STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES    = ("filepath", "metadata_before", "metadata_after", "metadata_json_before", "metadata_json_after",)
    OUTPUT_TOOLTIPS = (
        "Full path to the newly saved PNG file.",
        "Human-readable metadata from the source file before changes. Wire into Bond: Show Text.",
        "Human-readable metadata from the saved file after stamping. Wire into Bond: Show Text.",
        "Raw JSON metadata from the source file. Includes ComfyUI workflow data.",
        "Raw JSON metadata from the saved file. ComfyUI workflow/prompt data stripped.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images":          ("IMAGE",  {"tooltip": "The image tensor to save. Wire from VAE Decode or any image output."}),
            "source_filepath": ("STRING", {"default": "", "multiline": False, "placeholder": "Wire path from Bond: Load Image or Bond: Load Image From Path", "tooltip": "Path to the original source image. Used to read the before metadata."}),
            "filename_prefix": ("STRING", {"default": "output/bond/img", "tooltip": "Save path and filename prefix. e.g. 'output/bond/img' saves to output/bond/ named img_[timestamp].png. Use an absolute path like 'E:/photos/shoot' to save outside ComfyUI."}),
            "exiftool_path":   ("STRING", {"default": "exiftool", "multiline": False, "placeholder": "exiftool  or  C:\\exiftool\\exiftool.exe", "tooltip": "Path to exiftool. Leave as 'exiftool' if on your PATH, or provide full path e.g. C:\\exiftool\\exiftool.exe"}),
            **_camera_and_rights_inputs(),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def save_with_custom_metadata(
        self, images, source_filepath, filename_prefix, exiftool_path,
        camera_preset, override_make, override_model, override_lens,
        location_city, datetime_override, description, artist, copyright,
        keywords, custom_xmp_tags,
    ):
        exiftool_exe    = _resolve_exiftool(exiftool_path)
        preset_tags     = CAMERA_PRESETS.get(camera_preset, {})
        out_dir, prefix = _split_prefix(filename_prefix)
        os.makedirs(out_dir, exist_ok=True)
        output_root     = folder_paths.get_output_directory()

        # Step 1 — Read original metadata
        source = source_filepath.strip()
        if source and os.path.isfile(source):
            metadata_before      = f"📄 BEFORE — {os.path.basename(source)}\n{'─' * 36}\n{_read_metadata_summary(exiftool_exe, source)}"
            metadata_json_before = _read_metadata_json(exiftool_exe, source, strip_workflow=False)
        else:
            metadata_before      = "📄 BEFORE\n─────────────────────────────────────\n(no source filepath provided or file not found)"
            metadata_json_before = "{}"

        # Step 2 — Save PNG and stamp
        saved_paths, ui_images = [], []
        for i, img_tensor in enumerate(images):
            arr      = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix   = f"_{i:04d}" if len(images) > 1 else ""
            filename = f"{prefix}_{ts}{suffix}.png"
            filepath = os.path.join(out_dir, filename)
            Image.fromarray(arr, mode="RGB").save(filepath, format="PNG", compress_level=6)
            print(f"[BondSaveWithCustomMetadata] Saved: {filepath}")
            _run_exiftool(_common_exiftool_args(
                exiftool_exe, filepath, preset_tags,
                override_make, override_model, override_lens,
                location_city, datetime_override, description, artist, copyright,
                keywords, custom_xmp_tags, is_video=False,
            ))
            saved_paths.append(filepath)
            try:    subfolder = os.path.relpath(out_dir, output_root)
            except: subfolder = out_dir
            ui_images.append({"filename": filename, "subfolder": subfolder, "type": "output"})

        # Step 3 — Read back
        final            = saved_paths[-1]
        metadata_after      = f"✅ AFTER — {os.path.basename(final)}\n{'─' * 36}\n{_read_metadata_summary(exiftool_exe, final)}"
        metadata_json_after = _read_metadata_json(exiftool_exe, final, strip_workflow=True)

        filepath_out = final if len(saved_paths) == 1 else "\n".join(saved_paths)
        return {
            "ui":     {"images": ui_images},
            "result": (filepath_out, metadata_before, metadata_after, metadata_json_before, metadata_json_after),
        }


class BondSaveVideoWithMetadata:
    """
    Stamps EXIF, XMP, and QuickTime metadata onto an already-saved video file
    in-place using exiftool. Does not re-encode — only metadata is modified.
    Works with MP4, MOV, MKV, and most formats exiftool supports.

    Wire the filepath output from any video save node into video_path.
    The before outputs are empty strings now and will be populated when a
    Bond: Load Video node is added in a future update.

    Requires exiftool installed and the path set in the exiftool_path widget.
    """
    CATEGORY     = CAT_METADATA
    FUNCTION     = "stamp_metadata"
    OUTPUT_NODE  = True
    RETURN_TYPES    = ("STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES    = ("filepath", "metadata_before", "metadata_after", "metadata_json_before", "metadata_json_after",)
    OUTPUT_TOOLTIPS = (
        "The input video filepath passed through unchanged.",
        "Reserved for future use with Bond: Load Video. Currently returns an empty string.",
        "Human-readable metadata read back from the video after stamping. Wire into Bond: Show Text.",
        "Reserved for future use with Bond: Load Video. Currently returns an empty string.",
        "Raw JSON metadata from the video after stamping. ComfyUI workflow/prompt data stripped.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path":    ("STRING", {"forceInput": True, "tooltip": "Path to the video file to stamp. Wire from any video save node's filepath output."}),
            "exiftool_path": ("STRING", {"default": "exiftool", "multiline": False, "placeholder": "exiftool  or  C:\\exiftool\\exiftool.exe", "tooltip": "Path to exiftool. Leave as 'exiftool' if on your PATH."}),
            **_camera_and_rights_inputs(),
        }}

    def stamp_metadata(
        self, video_path, exiftool_path,
        camera_preset, override_make, override_model, override_lens,
        location_city, datetime_override, description, artist, copyright,
        keywords, custom_xmp_tags,
    ):
        filepath = video_path.strip()
        if not os.path.isfile(filepath): raise FileNotFoundError(f"Not found: {filepath}")
        exiftool_exe = _resolve_exiftool(exiftool_path)
        preset_tags  = CAMERA_PRESETS.get(camera_preset, {})
        _run_exiftool(_common_exiftool_args(
            exiftool_exe, filepath, preset_tags,
            override_make, override_model, override_lens,
            location_city, datetime_override, description, artist, copyright,
            keywords, custom_xmp_tags, is_video=True,
        ))
        metadata_after      = f"✅ AFTER — {os.path.basename(filepath)}\n{'─' * 36}\n{_read_metadata_summary(exiftool_exe, filepath)}"
        metadata_json_after = _read_metadata_json(exiftool_exe, filepath, strip_workflow=True)
        return (filepath, "", metadata_after, "", metadata_json_after)


# ===========================================================================
# NODE MAPPINGS
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "PromptJSONSelector":           PromptJSONSelector,
    "BatchIntPick":                 BatchIntPick,
    "BatchStringPick":              BatchStringPick,
    "BondPromptArrayIterator":      BondPromptArrayIterator,
    "CartesianIndexDriverImg":      CartesianIndexDriverImg,
    "CartesianIndexDriverImgPrmpt": CartesianIndexDriverImgPrmpt,
    "RangeStepper":                 RangeStepper,
    "BondSwitch2to1":               BondSwitch2to1,
    "BondSwitch1to2":               BondSwitch1to2,
    "BondLoadImage":                BondLoadImage,
    "BondLoadImageFromPath":        LoadImageFromPath,
    "BondLoadVideoFromPath":        LoadVideoFromPath,
    "BondBatchImageLoader":         BondBatchImageLoader,
    "BondShowText":                 BondShowText,
    "BondReadMetadata":             BondReadMetadata,
    "BondStripMetadata":            BondStripMetadata,
    "BondSaveWithCustomMetadata":   BondSaveWithCustomMetadata,
    "BondSaveVideoWithMetadata":    BondSaveVideoWithMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptJSONSelector":           "Bond: Prompt JSON/TXT Selector",
    "BatchIntPick":                 "Bond: Batch → Int Pick",
    "BatchStringPick":              "Bond: Batch → String Pick",
    "BondPromptArrayIterator":      "Bond: Prompt Array Iterator",
    "CartesianIndexDriverImg":      "Bond: Cartesian Index Driver (Image)",
    "CartesianIndexDriverImgPrmpt": "Bond: Cartesian Index Driver (Img + Prompt)",
    "RangeStepper":                 "Bond: Range Stepper",
    "BondSwitch2to1":               "Bond: Switch 2→1 🔀",
    "BondSwitch1to2":               "Bond: Switch 1→2 🔀",
    "BondLoadImage":                "Bond: Load Image 🖼️",
    "BondLoadImageFromPath":        "Bond: Load Image From Path",
    "BondLoadVideoFromPath":        "Bond: Load Video From Path",
    "BondBatchImageLoader":         "Bond: Batch Image Loader",
    "BondShowText":                 "Bond: Show Text 📝",
    "BondReadMetadata":             "Bond: Read Metadata 🔍",
    "BondStripMetadata":            "Bond: Strip Metadata 🧹",
    "BondSaveWithCustomMetadata":   "Bond: Save With Custom Metadata ✨",
    "BondSaveVideoWithMetadata":    "Bond: Save Video With Metadata 🎬",
}
