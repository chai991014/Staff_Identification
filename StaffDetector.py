import cv2
import csv
import os
import time
import argparse
from tqdm import tqdm
from ultralytics import YOLO


class StaffDetector:
    def __init__(self, video_path, person_model_path, nametag_model_path, output_dir, output_video_path, device=0):
        self.video_path = video_path
        self.person_model_path = person_model_path
        self.nametag_model_path = nametag_model_path
        self.output_dir = output_dir
        self.output_video_path = output_video_path
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)
        self.person_model = YOLO(self.person_model_path)
        self.nametag_model = YOLO(self.nametag_model_path)

    def run(self, frame_skip=5, person_conf=0.2, nametag_conf=0.3):

        coord_file = open(f"results/staff_coordinates_{frame_skip}_{nametag_conf}.csv", "w", newline="")
        csv_writer = csv.writer(coord_file)
        csv_writer.writerow(["frame_idx", "person_x1", "person_y1", "person_x2", "person_y2", "person_conf", "tag_x1", "tag_y1", "tag_x2", "tag_y2", "tag_conf"])

        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        saved_count = 0
        out_video = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=total_frames, desc="Processing video")

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if out_video is None:
                height, width = frame.shape[:2]
                out_video = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (width, height))

            if frame_idx % frame_skip == 0:
                clean_frame = frame.copy()
                person_results = self.person_model.predict(source=frame, device=self.device, verbose=False)[0]

                for i, box in enumerate(person_results.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == 0 and conf > person_conf:  # Class 0: person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_crop = clean_frame[y1:y2, x1:x2]

                        # Run nametag detection on the cropped person
                        nametag_results = self.nametag_model.predict(source=person_crop, device=self.device, verbose=False)[0]

                        for tag_box in nametag_results.boxes:
                            tag_conf = float(tag_box.conf[0])
                            if tag_conf > nametag_conf:
                                # Draw nametag box on cropped person image (relative coords)
                                tag_x1, tag_y1, tag_x2, tag_y2 = map(int, tag_box.xyxy[0])
                                cv2.rectangle(person_crop, (tag_x1, tag_y1), (tag_x2, tag_y2), (255, 0, 0), 2)

                                # Save cropped person with name tag box
                                filename = os.path.join(self.output_dir, f"staff_with_tag_{saved_count:04d}_{tag_conf:.2f}.jpg")
                                cv2.imwrite(filename, person_crop)
                                saved_count += 1

                                # Draw green box on original frame (person)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"Staff+Tag {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                # Draw blue box on original frame (nametag in absolute coords)
                                abs_x1 = x1 + tag_x1
                                abs_y1 = y1 + tag_y1
                                abs_x2 = x1 + tag_x2
                                abs_y2 = y1 + tag_y2
                                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"Tag {tag_conf:.2f}", (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                                csv_writer.writerow([frame_idx, x1, y1, x2, y2, conf, abs_x1, abs_y1, abs_x2, abs_y2, tag_conf])

            out_video.write(frame)
            frame_idx += 1
            progress.update(1)

        coord_file.close()
        cap.release()
        out_video.release()
        progress.close()

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        print(f"\n已保存带 name tag 的员工图像: {saved_count} 张")
        print(f"检测视频保存至: {self.output_video_path}")
        print(f"员工坐标 CSV 文件保存至: results/staff_coordinates_{frame_skip}_{nametag_conf}.csv")
        print(f"总运行时间: {int(minutes)} 分 {int(seconds)} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staff nametag detector")

    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--person_model_path', type=str, default="training/runs/detect/train/weights/best.pt", help='YOLO person detection model path')
    parser.add_argument('--nametag_model_path', type=str, default="training/runs/detect/yolo_nametag/weights/best.pt", help='YOLO nametag detection model path')
    parser.add_argument('--frame_skip', type=int, default=5, help='Process every Nth frame')
    parser.add_argument('--person_conf', type=float, default=0.5, help='Confidence threshold for person detection')
    parser.add_argument('--nametag_conf', type=float, default=0.75, help='Confidence threshold for nametag detection')
    parser.add_argument('--device', type=int, default=-1, help='CUDA device ID (e.g., 0) or -1 for CPU')

    args = parser.parse_args()

    nametag_conf_str = f"{args.nametag_conf:.2f}"
    output_dir = f"results/staff_with_nametag_{args.frame_skip}_{nametag_conf_str}"
    output_video_path = f"results/staff_output_with_boxes_{args.frame_skip}_{nametag_conf_str}.mp4"

    detector = StaffDetector(
        video_path=args.video_path,
        person_model_path=args.person_model_path,
        nametag_model_path=args.nametag_model_path,
        output_dir=output_dir,
        output_video_path=output_video_path,
        device=args.device
    )

    detector.run(
        frame_skip=args.frame_skip,
        person_conf=args.person_conf,
        nametag_conf=args.nametag_conf
    )
