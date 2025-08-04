# 🧍‍♂️ Staff Nametag Detection (Staff_Identification)

This project uses **YOLO11s** to automatically detect staff wearing nametags from video.

---

## 📌 Overview

- Detect people frame-by-frame from video
- Detect whether detected people are wearing nametags
- Crop and save images of staff wearing nametags
- Output video with detection boxes
- Generate CSV file with detection results

---

## 📂 Project Structure

```
Staff_Identification/
├── StaffDetector.py # Main detection script
├── temp.py # Temporary code
├── yolo11n.pt # YOLO nano pretrained weights
├── yolo11s.pt # YOLO small pretrained weights
├── training/ # Data and training related
│ ├── cropped_persons/ # Cropped person images
│ ├── dataset/ # YOLO training dataset (person)
│ ├── dataset_nametag/ # YOLO training dataset (nametag)
│ ├── raw_data/ # Raw video frames
│ ├── raw_data_labeled/ # Labeled data (person)
│ ├── raw_data_nametag/ # Raw nametag data
│ ├── raw_data_nametag_labeled/ # Labeled nametag data
│ ├── raw_data_persons/ # Detected person images
│ ├── runs/ # YOLO training output
│ ├── DataProcessing.py # Video frame extraction and dataset split
│ ├── output_with_boxes.mp4 # Output video with detection boxes
│ ├── yolo_ls.json # Label Studio config
│ ├── yolo_ls.label_config.xml # Label Studio label config
├── results/
│ ├── staff_with_nametag/ # Detected staff with nametags
│ ├── staff_coordinates_1_0.75.csv # Detection coordinates CSV
│ ├── staff_output_with_boxs_1_0.75.mp4 # Video with detection boxes
└── requirements.txt # Python dependencies
```

---

## ⚙️ Environment Setup

It is recommended to use **conda** to create an isolated environment:

```bash
conda create -n nametag python=3.10
conda activate nametag
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Preparation
Train person detection model (YOLO)
Place labeled person data in training/dataset/

Train nametag detection model
Place labeled nametag data in training/dataset_nametag/

If you already have trained models, place them in the project root:

```arduino
yolo11s.pt           # person detection model
yolo11n_nametag.pt   # nametag detection model
```
### 2. Extract Frames from Video
```bash
cd training
python DataProcessing.py --video_path path/to/video.mp4
```
This saves frames to training/raw_data/images/

### 3. Detect Staff and Nametags
```bash
python StaffDetector.py \
    --video_path path/to/video.mp4 \
    --person_model_path training/runs/detect/train/weights/best.pt \
    --nametag_model_path training/runs/detect/yolo_nametag/weights/best.pt \
    --output_dir results/ \
    --output_video_path results/staff_output_with_boxs.mp4 \
    --frame_skip 1 \
    --person_conf 0.2 \
    --nametag_conf 0.75
```
Main parameters:

--frame_skip: detect every N frames

--person_conf: confidence threshold for person detection

--nametag_conf: confidence threshold for nametag detection

---

## 📊 Output

CSV: staff_coordinates_*.csv
Coordinates of people in each frame

Video: staff_output_with_boxs_*.mp4
Video with detection boxes

Images: staff_with_nametag/
Cropped staff images with nametags

---

## 🧠 Technical Details

YOLO11s: person and nametag detection

OpenCV: video frame processing and drawing boxes

Pandas: managing detection results

---

## 📌 TODO

 Support real-time camera detection

 Few-shot learning for better performance

---

## 📜 License

This project is for research and educational purposes only. Commercial use is not allowed.