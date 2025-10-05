# 🕳️ Pothole Detection using YOLOv8

This project demonstrates how to train a lightweight YOLOv8 model to detect potholes and road cracks from images using annotated data in Pascal VOC (XML) format. It is optimized to run on low-resource hardware (e.g. Intel UHD 620 with 8GB RAM) using CPU-only training.

## 📁 Project Structure

```bash

.
├── annotated-images/              # Original images and their Pascal VOC (XML) annotations
├── splits.json                    # Train/test split metadata
├── xml2yolo_conversion.py         # Script to convert XML to YOLO format using splits.json
├── yolo_dataset/                  # Converted YOLOv8-compatible dataset
│   ├── data.yaml                  # YOLO dataset config
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── yolov8n.pt                     # YOLOv8n pre-trained weights
├── yolov8_pothole_detection.ipynb # Main training notebook
├── runs/                          # YOLOv8 training/inference output
├── pyproject.toml                 # Python project definition (used with uv)
├── uv.lock                        # uv lockfile
└── README.md                      # This file

````

## 🚀 Quick Start

### 1. Environment Setup

Install dependencies using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install ultralytics matplotlib opencv-python
````

> ✅ Note: No CUDA or NVIDIA drivers required. All training runs on CPU.

### 2. Convert Annotations (VOC → YOLO)

Make sure `splits.json` exists and includes the correct XML filenames for training/validation.

Run:

```bash
python xml2yolo_conversion.py
```

This creates the `yolo_dataset/` folder with YOLO-compatible images and labels.

### 3. Train YOLOv8n on CPU

Open the Jupyter notebook `yolov8_pothole_detection.ipynb` and run all cells.
This includes:

* Visualizing bounding boxes from the training set
* Training YOLOv8n (`nano` variant, CPU-friendly)
* Evaluating performance
* Running inference on validation images
* Plotting training metrics (loss, mAP, etc.)

### 4. Inference CLI Example

To run inference outside the notebook:

```bash
yolo task=detect \
  mode=predict \
  model=runs/detect/yolov8n_v8_5e/weights/best.pt \
  source=yolo_dataset/images/val \
  imgsz=640 \
  device=cpu \
  save_txt=True \
  hide_labels=True
```

Results are saved in `runs/detect/yolov8n_v8_5e_predict/`.

## 📊 Training Configuration

| Setting    | Value          |
| - | -- |
| Model      | YOLOv8n (Nano) |
| Epochs     | 5              |
| Image Size | 640 × 640      |
| Batch Size | 2              |
| Device     | CPU (no CUDA)  |

## 📚 Dataset Notes

Dataset source: [Annotated Potholes Image Dataset by Atikur Rahman Chitholian - Kaggle](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset)

* You can use the provided `splits.json` to divide XML-annotated images into train and validation sets.
* `data.yaml` is already configured and placed in `yolo_dataset/`.

## 🧑‍💻 Author

Built for experimentation and academic purposes with low-resource systems in mind.
Feel free to fork and contribute!
