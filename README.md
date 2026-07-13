# 🚗 Real-Time Road Scene Understanding for Autonomous Driving

An end-to-end **Computer Vision** pipeline for autonomous driving that combines **Semantic Segmentation**, **Object Detection**, and **Multi-Object Tracking** into a unified real-time perception system.

The project was developed using the **BDD100K (Berkeley DeepDrive 100K)** dataset and leverages **U-Net++**, **YOLO11m**, and **ByteTrack** to understand complex road scenes from driving videos.

---

## 📌 Project Overview

Modern autonomous vehicles require multiple perception tasks to be performed simultaneously. Instead of relying on a single deep learning model, this project integrates multiple specialized models into one pipeline.

The system performs:

- 🛣 Semantic segmentation of road scenes
- 🚘 Traffic object detection
- 🎯 Persistent multi-object tracking
- 🎥 Real-time video inference
- 📊 Live FPS monitoring

---

## 🎯 Features

- Semantic Segmentation using **U-Net++ (EfficientNet-B3 Encoder)**
- Object Detection using **YOLO11m**
- Multi-Object Tracking using **ByteTrack**
- Real-time video processing with OpenCV
- Drivable road highlighting
- Persistent object IDs across frames
- FPS calculation and visualization

---

# 🏗 Project Architecture

```
                   Input Video
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Semantic Segmentation            Object Detection
     (U-Net++)                     (YOLO11m)
        │                               │
        ▼                               ▼
 Road Segmentation Mask         Bounding Boxes
                                        │
                                        ▼
                                ByteTrack Tracker
                                        │
        └───────────────┬───────────────┘
                        ▼
            Visualization & Overlay
                        │
                        ▼
              Annotated Output Video
```

---

# 📂 Repository Structure

```
├── detect-drive.py          # YOLO training pipeline
├── segment-drive.py         # U-Net++ training pipeline
├── Main.py                  # Complete inference pipeline
├── yolo_best.pt             # Trained YOLO model
├── unetpp_effb3.pth         # Trained segmentation model
├── output_video.mp4         # Sample output
├── README.md
```

---

# 📚 Dataset

This project uses the **BDD100K (Berkeley DeepDrive 100K)** autonomous driving dataset.

The dataset contains:

- Road scenes
- Pixel-wise semantic labels
- Object detection annotations
- Diverse weather conditions
- Day & Night driving
- Urban and highway environments

Website:

https://bdd-data.berkeley.edu/

---

# 🧠 Models Used

## 1. Semantic Segmentation

Model:

- U-Net++

Encoder:

- EfficientNet-B3

Framework:

- Segmentation Models PyTorch

Number of classes:

- 19

Loss Function:

- Dice Loss
- Cross Entropy Loss

Optimizer:

- AdamW

Metric:

- Mean IoU

---

## 2. Object Detection

Model:

- YOLO11m

Framework:

- Ultralytics

Object Classes:

- Bike
- Bus
- Car
- Motor
- Person
- Rider
- Traffic Light
- Traffic Sign
- Train
- Truck

Optimizer:

- AdamW

Tracking:

- ByteTrack

---

# 📈 Performance

## Semantic Segmentation

| Metric | Score |
|---------|-------|
| Train mIoU | **61.17%** |
| Validation mIoU | **50.20%** |

---

## Object Detection

| Metric | Score |
|---------|-------|
| mAP@0.5 | **57.3%** |
| mAP@0.5:0.95 | **32.8%** |

---

# ⚙ Training Pipelines

## Semantic Segmentation

Implemented from scratch using PyTorch.

Pipeline includes:

- Dataset preparation
- Data augmentation
- Custom Dataset class
- DataLoader
- Dice + Cross Entropy Loss
- Mean IoU evaluation
- Learning rate scheduling
- Automatic checkpoint saving

---

## Object Detection

The detection pipeline performs:

- Parsing BDD100K annotations
- Converting annotations into YOLO format
- Dataset organization
- Automatic generation of dataset.yaml
- YOLO11m training using Ultralytics

---

# 🎥 Inference Pipeline

The inference pipeline performs the following operations for every frame:

1. Read input frame
2. Perform semantic segmentation
3. Perform object detection
4. Track detected objects using ByteTrack
5. Overlay drivable road segmentation
6. Draw tracked detections
7. Compute FPS
8. Save annotated frame

---

# 🛠 Technologies Used

- Python
- PyTorch
- Segmentation Models PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- Albumentations
- ByteTrack
- tqdm
- BDD100K

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
```

Move into the project

```bash
cd your-repository-name
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Usage

Run the inference pipeline

```bash
python Main.py
```

---

# 📷 Results

## Input

*(Add an input frame screenshot here.)*

---

## Semantic Segmentation

*(Add segmentation result here.)*

---

## Object Detection + Tracking

*(Add detection result here.)*

---

## Final Output

*(Add GIF or video preview here.)*

---

# 🔮 Future Improvements

- Lane detection integration
- Instance segmentation
- Depth estimation
- Traffic sign recognition
- Driver behavior analysis
- Sensor fusion with LiDAR
- Model optimization for embedded devices
- TensorRT deployment
- ONNX export
- Real-time edge deployment

---

# 📜 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Mubeen Akhund**

Software Engineering Undergraduate  
Mehran University of Engineering & Technology (MUET)

GitHub:
https://github.com/AkhundMubeen

LinkedIn:
(Add your LinkedIn profile)

---

## ⭐ If you found this project helpful, consider giving it a star!
