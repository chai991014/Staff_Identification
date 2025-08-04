# 🧍‍♂️ 员工名牌检测 (Staff_Identification)

本项目使用 **YOLO11s** 技术，从视频中自动检测佩戴名牌的员工。

---

## 📌 项目简介

- 从视频逐帧检测人员位置
- 检测检测到的人员是否佩戴名牌
- 裁剪并保存佩戴名牌的员工图像
- 输出带有检测框的视频
- 生成检测结果 CSV 文件

---

## 📂 项目结构

```
Staff_Identification/
├── StaffDetector.py # 主检测脚本
├── temp.py # 临时代码
├── yolo11n.pt # YOLO nano 预训练权重
├── yolo11s.pt # YOLO small 预训练权重
├── training/ # 数据与训练相关
│ ├── cropped_persons/ # 裁剪的人员图像
│ ├── dataset/ # YOLO 训练数据集（person）
│ ├── dataset_nametag/ # YOLO 训练数据集（nametag）
│ ├── raw_data/ # 原始视频帧
│ ├── raw_data_labeled/ # 已标注数据（person）
│ ├── raw_data_nametag/ # 原始 nametag 数据
│ ├── raw_data_nametag_labeled/ # 已标注 nametag 数据
│ ├── raw_data_persons/ # 检测到的人员图像
│ ├── runs/ # YOLO 训练输出
│ ├── DataProcessing.py # 视频帧提取与数据集划分
│ ├── output_with_boxes.mp4 # 带检测框的输出视频
│ ├── yolo_ls.json # Label Studio 配置
│ ├── yolo_ls.label_config.xml # Label Studio 标签配置
├── results/
│ ├── staff_with_nametag/ # 检测到 nametag 的员工图像
│ ├── staff_coordinates_1_0.75.csv # 检测坐标 CSV
│ ├── staff_output_with_boxs_1_0.75.mp4 # 带检测框的视频
└── requirements.txt # Python 依赖
```

---

## ⚙️ 环境安装

建议使用 **conda** 创建隔离环境：

```bash
conda create -n nametag python=3.10
conda activate nametag
pip install -r requirements.txt
```
---

## 🚀 使用方法

### 1. 数据准备
训练人员检测模型 (YOLO)
将标注好的人员数据放入 training/dataset/

训练名牌检测模型
将标注好的 nametag 数据放入 training/dataset_nametag/

如果已有训练好的模型，可直接放在项目根目录：
```arduino
yolo11s.pt           # person 检测模型
yolo11n_nametag.pt   # nametag 检测模型
```
### 2. 从视频中提取帧
```bash
cd training
python DataProcessing.py --video_path path/to/video.mp4
```
该脚本会将视频帧保存到 training/raw_data/images/
### 3. 检测员工与名牌
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
主要参数：

--frame_skip：每隔多少帧检测一次

--person_conf：person 检测的置信度阈值

--nametag_conf：nametag 检测的置信度阈值

---

## 📊 输出结果

CSV：staff_coordinates_*.csv
每帧的人员位置

视频：staff_output_with_boxs_*.mp4
带检测框的视频

图片：staff_with_nametag/
裁剪的佩戴名牌员工图片

---

## 🧠 技术细节

YOLO11s：人员和名牌检测

OpenCV：视频帧处理与绘制检测框

Pandas：结果数据管理

---

## 📌 TODO

 支持实时摄像头检测

 少样本学习提升检测效果

---

## 📜 许可证

本项目仅供研究与学习使用，请勿用于商业用途。
