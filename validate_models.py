import os
import gc
import torch
import pandas as pd
from ultralytics import YOLO

CONFIG_YAML = 'configs/config.yaml'

MODELS_TO_VALIDATE = {
    # "YOLO11n (Bez Augmentacji)": "runs/detect/PCB_AOI/yolo11_no_aug/weights/best.pt",
    # "YOLO11n (Z standardową Augmentacją)": "runs/detect/PCB_AOI/yolo11_standard_aug/weights/best.pt",
    "YOLO11n (Z customową Augmentacją)": "runs/detect/PCB_AOI/yolo11_custom_aug/weights/best.pt",
    # "YOLO26n (Bez Augmentacji)": "runs/detect/PCB_AOI/yolo26_no_aug/weights/best.pt",
    # "YOLO26n (Z standardową Augmentacją)": "runs/detect/PCB_AOI/yolo26_standard_aug/weights/best.pt",
    "YOLO26n (Z customową Augmentacją)": "runs/detect/PCB_AOI/yolo26_custom_aug/weights/best.pt"
}


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def calculate_f1(precision, recall):
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main():
    results_list = []

    for name, path in MODELS_TO_VALIDATE.items():
        if not os.path.exists(path):
            print(f"File not found '{name}'")
            continue

        model = YOLO(path)
        is_yolo26 = "yolo26" in name.lower()

        metrics = model.val(
            data=CONFIG_YAML,
            split='test',
            device=0,
            workers=2,
            plots=False,
            # nms=False if is_yolo26 else True
        )

        precision = metrics.results_dict['metrics/precision(B)']
        recall = metrics.results_dict['metrics/recall(B)']
        map50 = metrics.results_dict['metrics/mAP50(B)']
        map95 = metrics.results_dict['metrics/mAP50-95(B)']
        f1_score = calculate_f1(precision, recall)

        inference_ms = metrics.speed['inference']

        results_list.append({
            "Konfiguracja Eksperymentu": f"**{name}**",
            "mAP@0.5": f"{map50 * 100:.2f}%",
            "mAP@0.5:0.95": f"{map95 * 100:.2f}%",
            "Precision": f"{precision * 100:.2f}%",
            "Recall": f"{recall * 100:.2f}%",
            "F1-Score": f"{f1_score:.2f}",
            "Inference Time (ms)": f"{inference_ms:.2f} ms"
        })

        del model
        del metrics
        clear_gpu_memory()


    df = pd.DataFrame(results_list)
    markdown_table = df.to_markdown(index=False, tablefmt="pipe")


    print(markdown_table)


if __name__ == '__main__':
    main()