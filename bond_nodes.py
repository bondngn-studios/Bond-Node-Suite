import os
import json
import threading
import torch
import numpy as np
from PIL import Image

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

# --- CATEGORY CONSTANT ---
CAT = "Bond/Batch/Utilities"

# --- CLASSES ---

class PromptJSONSelector:
    _cache = {}  # path -> (mtime, list[str])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0}),
                "mode": (["auto", "json_array", "txt_lines"],),
                "delimiter": ("STRING", {"default": "\n"}),
                "wrap_index": (["wrap", "clamp"],),
            }
        }

    RETURN_TYPES, RETURN_NAMES = ("STRING", "INT"), ("prompt_text", "count")
    FUNCTION, CATEGORY = "run", CAT

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _load_list(path, mode, delimiter):
        if not os.path.isfile(path):
            return []
        mtime = os.path.getmtime(path)
        cache = PromptJSONSelector._cache.get(path)
        if cache and cache[0] == mtime:
            return cache[1]

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        lst = []
        if mode == "auto":
            try:
                data = json.loads(content)
                lst = [str(x) for x in data] if isinstance(data, list) else [line.strip() for line in content.splitlines() if line.strip()]
            except:
                lst = [line.strip() for line in content.splitlines() if line.strip()]
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("BATCH_INT", {}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
            },
            "optional": {"wrap": ("BOOLEAN", {"default": True})}
        }
    RETURN_TYPES, RETURN_NAMES = ("INT",), ("value",)
    FUNCTION, CATEGORY = "go", CAT

    def go(self, values, index, wrap=True):
        n = len(values) if values is not None else 0
        if n == 0: return (0,)
        i = (index % n) if wrap else max(0, min(index, n - 1))
        try: return (int(values[i]),)
        except: return (0,)

class BatchStringPick:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING", {"forceInput": True}), 
                "index": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
            },
            "optional": {"wrap": ("BOOLEAN", {"default": True})}
        }
    RETURN_TYPES, RETURN_NAMES = ("STRING",), ("text",)
    FUNCTION, CATEGORY, INPUT_IS_LIST = "go", CAT, True

    def go(self, texts, index, wrap=True):
        idx = index[0] if isinstance(index, list) else index
        do_wrap = wrap[0] if isinstance(wrap, list) else wrap
        if not texts: return ("",)
        if isinstance(texts[0], list): texts = texts[0]
        n = len(texts)
        i = (idx % n) if do_wrap else max(0, min(idx, n - 1))
        return (str(texts[i] if texts[i] is not None else ""),)

class BondPromptArrayIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_path": ("STRING", {"default": "path_to_your_file.json"}),
                "base_seed": ("INT", {"default": 123456, "min": 0, "max": 0x7FFFFFFF}),
                "advance_on_execution": ("BOOLEAN", {"default": True}),
                "pre_advance": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": False}),
                "reset": ("BOOLEAN", {"default": False})
            }
        }
    RETURN_TYPES, RETURN_NAMES = ("STRING", "INT", "INT", "BOOLEAN"), ("text", "index", "seed", "done")
    FUNCTION, CATEGORY = "run", CAT

    @classmethod
    def IS_CHANGED(cls, json_path, **kwargs):
        ap = _abs(json_path)
        st = _STATE.get(ap)
        idx = st["idx"] if st else -1
        return f"{ap}|idx={idx}|{json.dumps(kwargs)}"

    def run(self, json_path, base_seed=123456, advance_on_execution=True, pre_advance=True, loop=False, reset=False):
        ap, data = _load_json_array(json_path)
        with _STATE_LOCK:
            st = _STATE.get(ap)
            if st is None or reset: _STATE[ap] = {"idx": 0, "size": len(data), "data": data}
            st = _STATE[ap]
            size = st["size"]
            if size == 0: return ("", 0, base_seed, True)
            idx_to_emit = st["idx"]
            if advance_on_execution and pre_advance:
                st["idx"] = (idx_to_emit + 1) % size if loop else min(idx_to_emit + 1, size - 1)
            prompt, done, seed = st["data"][idx_to_emit], (idx_to_emit == size - 1) and (not loop), int(base_seed) + int(idx_to_emit)
            if advance_on_execution and not pre_advance:
                st["idx"] = (idx_to_emit + 1) % size if loop else min(idx_to_emit + 1, size - 1)
            return (prompt, idx_to_emit, seed, done)

class CartesianIndexDriverImg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"i": ("INT", {"default": 0}), "N1": ("INT", {"default": 1}), "N2": ("INT", {"default": 1}), "N3": ("INT", {"default": 1})}}
    RETURN_TYPES, RETURN_NAMES = ("INT","INT","INT","INT"), ("idx1","idx2","idx3","total")
    FUNCTION, CATEGORY = "run", CAT

    def run(self, i, N1, N2, N3):
        total = N1 * N2 * N3
        i = min(i, total - 1) if total > 0 else 0
        return (i // (N2 * N3), (i // N3) % N2, i % N3, total)

class CartesianIndexDriverImgPrmpt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "i": ("INT", {"default": 0}), "N1": ("INT", {"default": 1}), "N2": ("INT", {"default": 1}), "N3": ("INT", {"default": 1}), "Np": ("INT", {"default": 1}),
                "order": (["prompt_fastest", "prompt_slowest"],),
                "overflow_mode": (["clamp", "wrap"],),
            }
        }
    RETURN_TYPES, RETURN_NAMES = ("INT", "INT", "INT", "INT", "INT"), ("idx1", "idx2", "idx3", "idxP", "total")
    FUNCTION, CATEGORY = "run", CAT

    def run(self, i, N1, N2, N3, Np, order, overflow_mode):
        total = N1 * N2 * N3 * Np
        if total <= 0: return (0, 0, 0, 0, 0)
        i_eff = i % total if overflow_mode == "wrap" else max(0, min(i, total - 1))
        if order == "prompt_slowest": sS, sO, sB, sP = 1, N1, N1 * N2, N1 * N2 * N3
        else: sP, sB, sO, sS = 1, Np, N3 * Np, N2 * N3 * Np
        return ((i_eff // sS) % N1, (i_eff // sO) % N2, (i_eff // sB) % N3, (i_eff // sP) % Np, total)

class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "image.png"})}}
    RETURN_TYPES, RETURN_NAMES = ("IMAGE", "STRING", "STRING", "STRING"), ("image", "path", "stem", "dir")
    FUNCTION, CATEGORY = "load", CAT

    def load(self, file_path):
        ap = _abs(file_path)
        if not os.path.isfile(ap): raise FileNotFoundError(f"Not found: {ap}")
        return (_pil_to_tensor_rgb(Image.open(ap)), ap, os.path.splitext(os.path.basename(ap))[0], os.path.dirname(ap))

class RangeStepper:
    _state = {}
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"start": ("INT", {"default": 0}), "stop": ("INT", {"default": 0}), "step": ("INT", {"default": 1}), "reset": ("BOOLEAN", {"default": False})}, "optional": {"trigger": ("*",)}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("INT",), ("i",), "run", CAT

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def run(self, start, stop, step, reset=False, trigger=None):
        key = (start, stop, step)
        cur = start if reset or key not in self._state else self._state[key]
        out_i = max(start, min(cur, stop))
        self._state[key] = min(out_i + step, stop)
        return (out_i,)