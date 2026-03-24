# Bond Node Suite for ComfyUI

A collection of high-utility custom nodes for ComfyUI focused on batch processing, complex indexing, iterator logic, and image/video metadata management.

## 🚀 Installation
1. `cd custom_nodes`
2. `git clone https://github.com/bondngn-studios/Bond-Node-Suite/`
3. Restart ComfyUI.

> **Metadata nodes require [exiftool](https://exiftool.org/)** — download the Windows Executable, rename `exiftool(-k).exe` to `exiftool.exe`, and place it (along with the `exiftool_files/` folder) somewhere permanent. Set the `exiftool_path` widget on any metadata node to the full path, e.g. `C:\exiftool\exiftool.exe`.

---

## 🛠 Nodes

### Bond/Batch

#### Cartesian Index Driver (Image) & Cartesian Index Driver (Img + Prompt)
Map a master counter `i` to a multi-dimensional grid. Perfect for running every outfit against every background, or every prompt against every image.

#### Prompt JSON/TXT Selector
Designed to work with the Cartesian `idxP` output.
- **Modes:** Auto-detects JSON arrays (`["p1", "p2"]`) or TXT files (one prompt per line)
- **Control:** Wrap (loop back) or clamp (hold last) when the index exceeds the list size
- See `samplePrompts/` in the repo for correct formatting

#### Bond Prompt Array Iterator
Thread-safe iterator for JSON prompt arrays with built-in seed syncing and advance-on-execution logic.

#### Range Stepper
A stateful counter that auto-increments with every queue run.

#### Batch → Int Pick / Batch → String Pick
Pluck a single value from a batch by index.

---

### Bond/Utilities

#### Bond: Load Image 🖼️
Upload-based image loader with a **choose file to upload** button and image preview — just like the native ComfyUI Load Image node. Outputs `image`, `path`, `stem`, and `dir` for use with metadata nodes.

#### Bond: Load Image From Path
Load an image from a typed absolute file path. Best for local batch workflows where files are already on disk. Same outputs as Bond: Load Image.

#### Bond: Load Video From Path
Pass a video filepath into the workflow as a string. Wire the `path` output into a video-aware node (e.g. VHS Load Video) or into Bond: Save Video With Metadata.

#### Bond: Batch Image Loader
Point to a directory and load images one at a time. Supports `single_image` (by index), `sequential` (auto-advances each run), and `random` (seed-driven) modes.

#### Bond: Show Text 📝
Display any string output directly inside the node body. Wire any STRING into `text_in` — text renders inline with a monospace font and a subtle Bond watermark. Passes the string through as `text_out` for chaining. Text persists across page reloads.

---

### Bond/Metadata

> All metadata nodes require **exiftool** — see installation note above.

#### Bond: Save With Custom Metadata ✨
The all-in-one metadata node. Wire in an image and its source filepath, and it will:
1. Read and display the **original metadata** (before)
2. Save a fresh PNG to your specified output directory
3. Stamp new custom metadata onto it
4. Read back and display the **new metadata** (after)

Outputs `filepath`, `metadata_before`, `metadata_after`, and `metadata_summary` — wire these into Bond: Show Text nodes to see a live before/after comparison.

**Wiring:**
```
Bond: Load Image  (or Bond: Load Image From Path)
  ├── image  →  images
  └── path   →  source_filepath
```

**Camera Presets:** iPhone 15 Pro, iPhone 14 Pro, iPhone 13, Samsung Galaxy S24 Ultra, Google Pixel 8 Pro, Sony A7R V, Canon EOS R5, Nikon Z9, or Manual.

**Metadata fields:** Camera make/model/lens, GPS location (geocoded from city name via Nominatim), date/time, description, artist, copyright, keywords, and freeform XMP tags.

> ⚠️ Do **not** wire `dir` into `override_directory` unless you want to save back to the source image's folder. Leave it disconnected to use the `output_directory` widget.

#### Bond: Save Video With Metadata 🎬
Stamp EXIF/XMP/QuickTime metadata onto an already-saved video file in-place. Wire the filepath output from any video save node (e.g. VHS Save Video) into `video_path`. Supports MP4, MOV, MKV, and most formats exiftool handles.

#### Bond: Read Metadata 🔍
Read all metadata from any image or video file using exiftool. Outputs a formatted summary string and raw JSON. Use `common_only` filter for the most useful tags, or `all` to see everything. Filepath passes through for chaining.

#### Bond: Strip Metadata 🧹
Strip ALL metadata from an image or video file in-place. Optional backup before stripping. Filepath passes through for chaining into other nodes.

---

## 📁 Sample Files

- **`samplePrompts/`** — Sample JSON prompt file showing the correct format for Bond: Prompt Array Iterator
- **`Bond_Studios_Workflows/`** — Workflows from Bond Studios YouTube videos
- **`resources/`** — Includes one-off files that I might create as part of my videos
---

## 📺 YouTube & Community
Workflows from YouTube videos are in the `Bond_Studios_Workflows/` folder in this repo.
