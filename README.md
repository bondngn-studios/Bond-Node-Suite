# Bond Node Suite for ComfyUI

A collection of high-utility custom nodes for ComfyUI focused on batch processing, complex indexing, and iterator logic.

## 🛠 Nodes Included
All nodes are under: `Bond ➔ Batch ➔ Utilities`

### 1. Cartesian Index Drivers
Available in **Image Only** and **Image + Prompt**. Map a master counter to a multi-dimensional grid. Perfect for "running every outfit against every background."

### 2. Prompt JSON/TXT Selector (NEW)
Specifically designed to work with the Cartesian `idxP` output.
- **Modes:** Auto-detects between JSON arrays (`["p1", "p2"]`) or TXT files (one prompt per line).
- **Control:** Choose to "wrap" (start over) or "clamp" (stay on the last prompt) when the index exceeds the list size.
- **Reference:** Check the `sample_prompts.json` included in this repo for the correct formatting.

### 3. Bond Prompt Array Iterator
A thread-safe node for iterating through JSON strings with built-in seed syncing and "advance on execution" logic.

### 4. Range Stepper
A stateful counter that auto-increments with every queue run.

### 5. Batch Pickers & Loaders
- **Batch → Int/String Pick**: Pluck values from batches by index.
- **Load Image From Path**: Load local images with full metadata output (stem, path, directory).

### 6. Batch Image Loader
- **Bond: Batch Image Loader**: Allows you to point to a directory and load images.  Select your path and set the number (next to Run) for how many images are in your folder.

## 🚀 Installation
1. `cd custom_nodes`
2. `git clone https://github.com/bondngn-studios/Bond-Node-Suite/`
3. Restart ComfyUI.


### Sample Files
In the repo, I've created a "samplePrompts" folder that will give you a sample file for how the JSON prompt that feeds the Bond Prompt Array Iterator should be formatted.

### YouTube & Community Workflows
In the repo, I've created a "Bond_Studios_Workflows" folder that contains all of the workflows from my YouTube Videos.
