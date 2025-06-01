import os
import shutil
import json
import xml.etree.ElementTree as ET

# ============================
# 🔧 KONFIGURASI
# ============================
ANNOTATIONS_DIR = "annotated-images"
IMAGES_DIR = "annotated-images"  # gambar dan XML di folder yang sama
SPLIT_JSON_PATH = "splits.json"

YOLO_DATASET_DIR = "yolo_dataset"
CLASS_NAMES = ["pothole"]  # <- ubah sesuai class di dataset kamu

# ============================
# 🏗️ Membuat Struktur Folder
# ============================
def create_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, "labels", split), exist_ok=True)

# ============================
# 📦 Konversi XML ke YOLO format
# ============================
def convert_xml_to_yolo(xml_file, class_names):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_labels = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in class_names:
            continue  # abaikan label yang tidak dikenal
        cls_id = class_names.index(cls)

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Konversi ke format YOLO
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_labels

# ============================
# 🔄 Proses per file
# ============================
def process_split(split_name, xml_list):
    for xml_file in xml_list:
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text
        image_src_path = os.path.join(IMAGES_DIR, filename)

        # Konversi anotasi
        yolo_lines = convert_xml_to_yolo(xml_path, CLASS_NAMES)

        # Simpan file label YOLO
        label_dest = os.path.join(YOLO_DATASET_DIR, "labels", split_name, filename.replace(".jpg", ".txt"))
        with open(label_dest, "w") as f:
            f.write("\n".join(yolo_lines))

        # Salin gambar
        image_dest = os.path.join(YOLO_DATASET_DIR, "images", split_name, filename)
        shutil.copy(image_src_path, image_dest)

# ============================
# 🚀 Eksekusi
# ============================
if __name__ == "__main__":
    print("📂 Membaca splits.json ...")
    with open(SPLIT_JSON_PATH) as f:
        splits = json.load(f)

    print("📁 Membuat struktur folder YOLO ...")
    create_dirs()

    print("🔄 Proses training set ...")
    process_split("train", splits["train"])

    print("🔄 Proses validation set ...")
    process_split("val", splits["test"])

    print("✅ Konversi selesai. Siap untuk training!")
