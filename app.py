import os
import gradio as ui
from ultralytics import YOLO

AVAILABLE_MODELS = {
    "YOLO11n (Z customową Augmentacją)": "runs/detect/PCB_AOI/yolo11_custom_aug/weights/best.pt",
    "YOLO26n (Z customową Augmentacją)": "runs/detect/PCB_AOI/yolo26_custom_aug/weights/best.pt",
    "YOLO11n (Bez Augmentacji)": "runs/detect/PCB_AOI/yolo11_no_aug/weights/best.pt",
    "YOLO11n (Z standardową Augmentacją)": "runs/detect/PCB_AOI/yolo11_standard_aug/weights/best.pt",
    "YOLO26n (Bez Augmentacji)": "runs/detect/PCB_AOI/yolo26_no_aug/weights/best.pt",
    "YOLO26n (Z standardową Augmentacją)": "runs/detect/PCB_AOI/yolo26_standard_aug/weights/best.pt"
}


def predict_pcb_defects(source_image, model_name, confidence_threshold):
    if source_image is None:
        return None

    model_path = AVAILABLE_MODELS[model_name]

    if not os.path.exists(model_path):
        return None

    model = YOLO(model_path)

    is_yolo26 = "yolo26" in model_name.lower()

    results = model.predict(
        source=source_image,
        conf=confidence_threshold,
        device="cpu",
        agnostic_nms=False if is_yolo26 else False
    )

    result = results[0]

    annotated_image = result.plot()

    return annotated_image


# Budowanie interfejsu graficznego Gradio
with ui.Blocks(title="PCB AOI - Defect Detection System") as app:
    ui.Markdown("# System Automatycznej Inspekcji Optycznej (AOI) PCB")
    ui.Markdown("Wgraj zdjęcie obwodu drukowanego, aby wykryć i sklasyfikować defekty struktury miedzi.")

    with ui.Row():
        with ui.Column(scale=1):
            # Panel sterowania (lewa strona)
            model_dropdown = ui.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value="YOLO11n (Z customową Augmentacją)",
                label="Wybierz architekturę i konfigurację modelu"
            )

            conf_slider = ui.Slider(
                minimum=0.01,
                maximum=1.0,
                value=0.25,
                step=0.01,
                label="Confidence Threshold"
            )

            input_img = ui.Image(type="numpy", label="Wejściowe zdjęcie PCB")
            btn_submit = ui.Button("Uruchom detekcję wad", variant="primary")

        with ui.Column(scale=1):
            output_img = ui.Image(type="numpy", label="Wynik inspekcji (Wykryte anomalie)")

    btn_submit.click(
        fn=predict_pcb_defects,
        inputs=[input_img, model_dropdown, conf_slider],
        outputs=output_img
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)