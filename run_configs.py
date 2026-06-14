
BASE_PARAMS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "device": 0,
    "workers": 2,
    "patience": 15,
    "amp": True,
    "project": "PCB_AOI"
}

RAW_MODEL_BASE_PARAMS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "device": 0,
    "workers": 2,
    "patience": 15,
    "Pretrained": False,
    "amp": True,
    "project": "PCB_AOI"
}

NO_AUGMENTATION_OVERRIDES = {
    "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0, "scale": 0.0,
    "translate": 0.0, "shear": 0.0, "perspective": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
    "auto_augment": None,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0
}

AOI_SPECIALIZED_AUGMENTATIONS = {
    "degrees": 90.0,
    "fliplr": 0.5,
    "flipud": 0.5,
    "translate": 0.05,

    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "copy_paste": 0.0,

    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.1,
    "bgr": 0.0,
    "auto_augment": None,

    "mosaic": 1.0,
    "mixup": 0.0,
    "erasing": 0.2,
}


YOLO11_NO_AUG = {**BASE_PARAMS, **NO_AUGMENTATION_OVERRIDES, "name": "yolo11_no_aug"}

YOLO26_NO_AUG = {**BASE_PARAMS, **NO_AUGMENTATION_OVERRIDES, "name": "yolo26_no_aug"}

YOLO11_WITH_AUG = {**BASE_PARAMS, "name": "yolo11_standard_aug"}

YOLO26_WITH_AUG = {**BASE_PARAMS, "name": "yolo26_standard_aug"}

YOLO26_WITH_CUSTOM_AUG = {**BASE_PARAMS, **AOI_SPECIALIZED_AUGMENTATIONS, "name": "yolo26_custom_aug"}

YOLO11_WITH_CUSTOM_AUG = {**BASE_PARAMS, **AOI_SPECIALIZED_AUGMENTATIONS, "name": "yolo11_custom_aug"}
