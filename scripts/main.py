import os
import cv2
import shutil
import matplotlib.pyplot as plt
from collections import Counter
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage.color import rgb2gray
import joblib

# === PATH SETTING ===
DATASET_PATH = '../homeObject'
OUTPUT_PATH = '../datasetCrop'

# === STEP 1: EKSTRAKSI CITRA PER OBJEK ===
def extract_objects():
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    def process_split(split):
        image_dir = os.path.join(DATASET_PATH, split, 'images')
        label_dir = os.path.join(DATASET_PATH, split, 'labels')
        
        for fname in tqdm(os.listdir(image_dir)):
            if not fname.endswith('.jpg'):
                continue
            image_path = os.path.join(image_dir, fname)
            label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))

            img = cv2.imread(image_path)
            if img is None:
                continue
            h, w, _ = img.shape

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x, y, bw, bh = map(float, parts)
                    class_id = int(class_id)

                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)

                    crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

                    class_folder = os.path.join(OUTPUT_PATH, split, str(class_id))
                    os.makedirs(class_folder, exist_ok=True)
                    crop_filename = f"{fname.replace('.jpg','')}_{i}.jpg"
                    cv2.imwrite(os.path.join(class_folder, crop_filename), crop)

    for split in ['train', 'valid']:
        process_split(split)
    print("✅ Ekstraksi objek selesai.")

# === STEP 2A: EDA SEBELUM BALANCING ===
def run_eda():
    print("📊 Menjalankan EDA (SEBELUM BALANCING)...")
    base_path = os.path.join(OUTPUT_PATH, 'train')
    class_counts = {}

    for class_id in os.listdir(base_path):
        n = len(os.listdir(os.path.join(base_path, class_id)))
        class_counts[class_id] = n

    plt.figure(figsize=(10,5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Distribusi Citra per Class (SEBELUM BALANCING)")
    plt.xlabel("Class ID")
    plt.ylabel("Jumlah Citra")
    plt.tight_layout()
    plt.show()

    # Contoh gambar
    fig, axs = plt.subplots(2, 5, figsize=(15,6))
    axs = axs.flatten()
    for idx, class_id in enumerate(sorted(os.listdir(base_path))[:10]):
        img_path = glob(f"{base_path}/{class_id}/*.jpg")[0]
        img = Image.open(img_path)
        axs[idx].imshow(img)
        axs[idx].axis("off")
        axs[idx].set_title(f"Class {class_id}")
    plt.tight_layout()
    plt.show()

# === STEP 2B: EDA SESUDAH BALANCING ===
def run_eda_balance():
    print("📊 Menjalankan EDA (SESUDAH BALANCING)...")
    base_path = os.path.join('../datasetCrop_balanced/train')
    class_counts = {}

    for class_id in os.listdir(base_path):
        n = len(os.listdir(os.path.join(base_path, class_id)))
        class_counts[class_id] = n

    plt.figure(figsize=(10,5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Distribusi Citra per Class (SESUDAH BALANCING)")
    plt.xlabel("Class ID")
    plt.ylabel("Jumlah Citra")
    plt.tight_layout()
    plt.show()

    # Contoh gambar
    fig, axs = plt.subplots(2, 5, figsize=(15,6))
    axs = axs.flatten()
    for idx, class_id in enumerate(sorted(os.listdir(base_path))[:10]):
        img_path = glob(f"{base_path}/{class_id}/*.jpg")[0]
        img = Image.open(img_path)
        axs[idx].imshow(img)
        axs[idx].axis("off")
        axs[idx].set_title(f"Class {class_id}")
    plt.tight_layout()
    plt.show()


# === STEP 3: TRAINING KLASIFIKASI ===
def extract_features(folder):
    X, y = [], []
    for class_id in os.listdir(folder):
        class_folder = os.path.join(folder, class_id)
        for fname in os.listdir(class_folder):
            img_path = os.path.join(class_folder, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Ekstraksi fitur HOG
            feature = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            X.append(feature)
            y.append(int(class_id))
    return np.array(X), np.array(y)

def train_model():
    print("🧠 Training Naive Bayes classifier...")
    X_train, y_train = extract_features(os.path.join('../datasetCrop_balanced/train'))
    X_val, y_val = extract_features(os.path.join(OUTPUT_PATH, 'valid'))

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("📋 Classification Report:")
    print(classification_report(y_val, y_pred))

    # === Simpan model dan class_map ===
    import joblib

    # Buat mapping label ke nama class
    class_map = {
        0: "bed", 1: "sofa", 2: "chair", 3: "table",
        4: "lamp", 5: "tv", 6: "laptop", 7: "wardrobe",
        8: "window", 9: "door", 10: "potted plant", 11: "photo frame"
    }

    joblib.dump(model, "naivebayes_model.pkl")
    joblib.dump(class_map, "class_map.pkl")

    print("✅ Model dan class_map disimpan.")

def balance_dataset(input_train_path, output_balanced_train_path):
    import random
    print(f"🔄 Balancing dataset from {input_train_path} to {output_balanced_train_path}...")

    if os.path.exists(output_balanced_train_path):
        shutil.rmtree(output_balanced_train_path)
    os.makedirs(output_balanced_train_path, exist_ok=True)

    class_counts = {}
    for class_id_folder in os.listdir(input_train_path):
        class_path = os.path.join(input_train_path, class_id_folder)
        if os.path.isdir(class_path):
            class_counts[class_id_folder] = len(os.listdir(class_path))

    if not class_counts:
        print("❌ Tidak ada kelas ditemukan untuk balancing. Pastikan path input benar.")
        return

    max_samples = max(class_counts.values())
    print(f"Target sampel per kelas setelah balancing: {max_samples}")

    for class_id_folder in os.listdir(input_train_path):
        class_path = os.path.join(input_train_path, class_id_folder)
        if not os.path.isdir(class_path): continue

        output_class_path = os.path.join(output_balanced_train_path, class_id_folder)
        os.makedirs(output_class_path, exist_ok=True)

        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        current_samples = len(images)

        for img_name in images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(output_class_path, img_name))

        if current_samples < max_samples:
            num_to_add = max_samples - current_samples
            print(f"  Class {class_id_folder}: {current_samples} -> {max_samples} (Menambahkan {num_to_add} sampel)")
            for i in range(num_to_add):
                if images:
                    img_to_duplicate = random.choice(images)
                    original_path = os.path.join(class_path, img_to_duplicate)
                    new_name = f"augmented_{i}_{img_to_duplicate}"
                    shutil.copy(original_path, os.path.join(output_class_path, new_name))
                else:
                    print(f"    Peringatan: Tidak ada gambar di kelas {class_id_folder} untuk diduplikasi.")

    print("✅ Balancing dataset selesai.")



# === MAIN ===
if __name__ == "__main__":
    extract_objects()
    run_eda()
    balance_dataset("../datasetCrop/train", "../datasetCrop_balanced/train")
    run_eda_balance()
    train_model()
