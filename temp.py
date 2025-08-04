import cv2
import os
from ultralytics import YOLO


class StaffDetector:
    def __init__(self, video_path, person_model_path, output_dir="cropped_persons", output_video_path="output_with_boxes.mp4", device=0):
        self.video_path = video_path
        self.person_model_path = person_model_path
        self.output_dir = output_dir
        self.output_video_path = output_video_path
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)
        self.model = YOLO(self.person_model_path)

    def detect_person(self, frame_skip=5, conf_threshold=0.1):
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        saved_count = 0
        out_video = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if out_video is None:
                height, width = frame.shape[:2]
                out_video = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (width, height))

            if frame_idx % frame_skip == 0:
                results = self.model.predict(source=frame, device=self.device)[0]

                clean_frame = frame.copy()

                for i, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == 0 and conf > conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

                        # Crop person image
                        x1_crop = max(x1, 0)
                        y1_crop = max(y1, 0)
                        x2_crop = min(x2, frame.shape[1])
                        y2_crop = min(y2, frame.shape[0])
                        crop = clean_frame[y1_crop:y2_crop, x1_crop:x2_crop]

                        if crop.shape[0] > 20 and crop.shape[1] > 20:
                            filename = os.path.join(self.output_dir, f"person_{saved_count:04d}.jpg")
                            cv2.imwrite(filename, crop)
                            saved_count += 1

            out_video.write(frame)
            frame_idx += 1

        cap.release()
        out_video.release()
        print(f"✅ 共保存 person 图像: {saved_count} 张")
        print(f"🎥 可视化视频保存至: {self.output_video_path}")


if __name__ == "__main__":
    detector = StaffDetector(
        video_path="sample.mp4",
        person_model_path="runs/detect/train/weights/best.pt",
        output_dir="cropped_persons",
        output_video_path="output_with_boxes.mp4",
        device=0
    )
    detector.detect_person(frame_skip=5)
