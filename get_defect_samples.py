import os
import cv2

IMAGES_DIR = r"data\test\images"
LABELS_DIR = r"data\test\labels"
OUTPUT_DIR = r"research"

CLASS_NAMES = {
    0: "copper",
    1: "mousebite",
    2: "open",
    3: "pin-hole",
    4: "short",
    5: "spur"
}

CROP_SIZE = 150
HALF_CROP = CROP_SIZE // 2


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved_classes = set()
    total_classes_to_find = len(CLASS_NAMES)

    print("Rozpoczynanie skanowania datasetu w poszukiwaniu próbek klas...")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(image_extensions)]

    for img_file in image_files:
        if len(saved_classes) == total_classes_to_find:
            break

        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"

        label_path = os.path.join(LABELS_DIR, label_file)
        image_path = os.path.join(IMAGES_DIR, img_file)

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        h_img, w_img, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if class_id in saved_classes:
                continue

            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])

            cx = int(x_center_norm * w_img)
            cy = int(y_center_norm * h_img)

            x_start = cx - HALF_CROP
            y_start = cy - HALF_CROP
            x_end = cx + HALF_CROP
            y_end = cy + HALF_CROP

            if x_start < 0:
                x_end -= x_start
                x_start = 0
            if y_start < 0:
                y_end -= y_start
                y_start = 0
            if x_end > w_img:
                x_start -= (x_end - w_img)
                x_end = w_img
            if y_end > h_img:
                y_start -= (y_end - h_img)
                y_end = h_img

            x_start, y_start = max(0, x_start), max(0, y_start)

            crop = img[y_start:y_end, x_start:x_end]

            class_name = CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")
            out_filename = f"class_{class_id}_{class_name}.png"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            cv2.imwrite(out_path, crop)

            print(f" Sukces: Wycięto i zapisano klasę {class_id} ({class_name}) -> {out_filename}")
            saved_classes.add(class_id)
            break

    print("\n=== PODSUMOWANIE PROCESU ===")
    print(f"Znaleziono i zapisano: {len(saved_classes)} z {total_classes_to_find} klas.")
    if len(saved_classes) < total_classes_to_find:
        missing = set(CLASS_NAMES.keys()) - saved_classes
        missing_names = [CLASS_NAMES[m] for m in missing]
        print(f"⚠️ Brakujące klasy w przeszukiwanym folderze: {missing_names}")
    print(f"Wszystkie pliki znajdziesz w katalogu: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()