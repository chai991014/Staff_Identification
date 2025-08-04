import cv2
import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO


def split_train_val(source_dir="raw_data_labeled", target_dir="dataset", val_ratio=0.2):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"

    image_paths = list(image_dir.glob("*.jpg"))
    random.shuffle(image_paths)

    split_index = int(len(image_paths) * (1 - val_ratio))
    train_imgs = image_paths[:split_index]
    val_imgs = image_paths[split_index:]

    for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out_dir = target_dir / "images" / subset
        label_out_dir = target_dir / "labels" / subset
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            label_path = label_dir / f"{img_path.stem}.txt"
            shutil.copy(img_path, img_out_dir / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, label_out_dir / label_path.name)

    print(f"✅ 数据集划分完成：train={len(train_imgs)}, val={len(val_imgs)}")


class DataProcessing:
    def __init__(self, video_path, output_dir="raw_data", dataset_dir="dataset", val_ratio=0.2):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.dataset_dir = Path(dataset_dir)
        self.val_ratio = val_ratio

    def extract_frames(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path)

        frame_interval = 7
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                filename = self.output_dir / "images" / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"已提取 {saved_count} 帧到 {self.output_dir}")

    def auto_label_person(self, model_path="yolov8s.pt", conf=0.25):
        model = YOLO(model_path)
        img_dir = self.output_dir / "images"
        label_dir = self.output_dir / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.glob("*.jpg"):
            results = model.predict(source=img_path, conf=conf, device=0)[0]
            labels = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = r
                if int(cls) != 0:
                    continue
                img = cv2.imread(str(img_path))
                h, w, _ = img.shape
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            label_path = label_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

        print("对全部图像完成 person 检测并生成标签")


if __name__ == "__main__":
    # preparer = DataProcessing(video_path="../sample.mp4")
    # preparer.extract_frames()
    # preparer.auto_label_person(model_path="../yolo11s.pt", conf=0.3)
    # split_train_val(source_dir="raw_data_labeled", target_dir="dataset")
    split_train_val(source_dir="raw_data_nametag_labeled", target_dir="dataset_nametag")

# train model cmd
# yolo detect train model=yolo11s.pt data=training/dataset/data.yaml epochs=100 imgsz=640 batch=16 device=0 augment=True
# yolo detect train model=yolo11s.pt data=training/dataset_nametag/data.yaml epochs=100 imgsz=416 batch=16 name='yolo_nametag' device=0 augment=True
