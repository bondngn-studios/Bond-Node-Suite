# Bond Node Suite for ComfyUI

A collection of high-utility custom nodes for ComfyUI focused on batch processing, complex indexing, and iterator logic. Designed to make large-scale generations and multi-variable testing (Subjects, Outfits, Backgrounds) much easier to manage.

## 🛠 Nodes Included

All nodes are located under the category: `Bond ➔ Batch ➔ Utilities`

### 1. Cartesian Index Drivers
Available in two flavors (**Image Only** and **Image + Prompt**). These nodes take a master counter (like a Frame Index or Range Stepper) and map it to a multi-dimensional grid. Perfect for "running every outfit against every background" automatically.

### 2. Bond Prompt Array Iterator
A thread-safe node that reads a JSON array of strings and iterates through them. 
- **Features:** Advance on execution, auto-looping, and a built-in seed generator that stays synced with the current index.

### 3. Range Stepper
A stateful counter that increments every time the queue runs. Use this to drive the "i" input on the Cartesian Drivers or any index-based node.

### 4. Batch Pickers (Int & String)
Allows you to pluck a single value out of a batch or list by index.
- **Batch → Int Pick**: Useful for dynamic seed or parameter selection.
- **Batch → String Pick**: Useful for choosing specific prompts or attributes from a list.

### 5. Load Image From Path
A simple but powerful utility to load an image from a direct file path on your hard drive, outputting the image tensor along with its filename and directory metadata.

## 🚀 Installation

1. Navigate to your ComfyUI `custom_nodes` folder.
2. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Bond-Node-Suite.git](https://github.com/YOUR_USERNAME/Bond-Node-Suite.git)


### Sample Files
In the repo, I've created a "samplePrompts" folder that will give you a sample file for how the JSON prompt that feeds the Bond Prompt Array Iterator should be formatted.