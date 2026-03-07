from .bond_nodes import (
    BatchIntPick, 
    BatchStringPick, 
    BondPromptArrayIterator, 
    CartesianIndexDriverImg, 
    CartesianIndexDriverImgPrmpt, 
    LoadImageFromPath, 
    RangeStepper,
    PromptJSONSelector,
    BondBatchImageLoader,          # <-- add this
)

NODE_CLASS_MAPPINGS = {
    "BatchIntPick": BatchIntPick,
    "BatchStringPick": BatchStringPick,
    "BondPromptArrayIterator": BondPromptArrayIterator,
    "CartesianIndexDriverImg": CartesianIndexDriverImg,
    "CartesianIndexDriverImgPrmpt": CartesianIndexDriverImgPrmpt,
    "LoadImageFromPath": LoadImageFromPath,
    "RangeStepper": RangeStepper,
    "PromptJSONSelector": PromptJSONSelector,
    "BondBatchImageLoader": BondBatchImageLoader,          # <-- add this
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchIntPick": "Bond: Batch → Int Pick",
    "BatchStringPick": "Bond: Batch → String Pick",
    "BondPromptArrayIterator": "Bond: Prompt Array Iterator",
    "CartesianIndexDriverImg": "Bond: Cartesian Index Driver (Image)",
    "CartesianIndexDriverImgPrmpt": "Bond: Cartesian Index Driver (Img + Prompt)",
    "LoadImageFromPath": "Bond: Load Image From Path",
    "RangeStepper": "Bond: Range Stepper",
    "PromptJSONSelector": "Bond: Prompt JSON/TXT Selector",
    "BondBatchImageLoader": "Bond: Batch Image Loader",    # <-- add this
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]