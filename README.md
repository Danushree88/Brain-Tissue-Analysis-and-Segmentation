# 🧠 Brain Tissue Analysis and Segmentation

> **End-to-end deep learning pipeline for MRI brain tissue segmentation, volumetric biomarker extraction, disease risk scoring, and tumor classification.**

**Authors:** Danushree R S (23PD05) · Harini Sree J (23PD14)

---

## 📄 Abstract

Brain tissue segmentation and volumetric analysis from MRI scans are critical steps in neurological disease diagnosis, yet they remain time-consuming and operator-dependent when performed manually. This project presents an end-to-end deep learning pipeline that automates the complete workflow — from raw T1-weighted MRI input to clinical-grade outputs. A custom 2D U-Net with skip connections is trained on the IBSR dataset (20 subjects) using a combined weighted Dice and cross-entropy loss with median-frequency class balancing, achieving Dice scores of 0.914 (GM), 0.851 (WM), and 0.861 (CSF) on the validation set. Predicted segmentation masks are used to extract six volumetric biomarkers including Brain Parenchymal Fraction and GM/WM ratio, which are compared against normal reference ranges to generate a disease support score. Subjects scoring above 35 are routed to a tumor classification module trained on the BraTS Kaggle dataset (7,200 images, 4 classes), where VGG16 fine-tuning achieves 94.3% test accuracy with a macro F1 of 0.941. The complete pipeline achieves 100% accuracy across a four-sample end-to-end test. An interactive Streamlit dashboard enables real-time MRI upload, segmentation visualization, biomarker display, and tumor prediction.

**Keywords:** Brain MRI · U-Net · Tissue Segmentation · Volumetric Biomarkers · VGG16 · Tumor Classification · BraTS · IBSR

---

## 📁 Repository Structure

```
Brain_Tissue_Analysis_and_Segmentation/
│
├── Brain_Tissue_Analysis_and_Segmentation_(6).ipynb   # Main notebook (all sections)
├── app.py                                              # Streamlit dashboard
├── README.md                                           # This file
│
├── requirements.txt                                    # Python dependencies
│
└── assets/                                             # (optional) screenshots
    ├── pipeline_overview.png
    ├── segmentation_results.png
    └── tumor_classification.png
```

---

## 🗂️ Notebook Sections

The notebook (`Brain_Tissue_Analysis_and_Segmentation_(6).ipynb`) covers the full pipeline in one file:

| Section | Description |
|---|---|
| **1. Dataset Overview** | Load IBSR `.npy` volumes, visualise slices and masks, audit class distribution |
| **2. Model Training** | 2D U-Net definition, combined loss, data augmentation, training loop |
| **3. Biomarker Extraction** | Volume computation (GM/WM/CSF/BPF/GM-WM ratio) per patient |
| **4. Disease Support Scoring** | Z-score deviation from normal ranges → 0–100 risk score |
| **5. Tumor Classification** | DenseNet201+PCA+SVM, VGG16 fine-tuning, EfficientNetB3 on BraTS |
| **6. Streamlit App** | Interactive dashboard code embedded and launched via localtunnel/cloudflared |

---

## 📊 Datasets

### IBSR (Internet Brain Segmentation Repository)
- 20 healthy adult subjects, 3D T1-weighted MRI
- Volume shape: `(48, 192, 192, 1)`
- 4-channel one-hot masks: Background · CSF · GM · WM
- Split: 14 train / 6 validation
- Download: [IBSR on Nitrc](https://www.nitrc.org/projects/ibsr)

### BraTS Kaggle Dataset
- 7,200 JPG brain MRI images
- 4 classes: Glioma · Meningioma · No Tumor · Pituitary
- Split: 5,600 train / 1,600 test (~1,400 per class)
- Download: [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

> **Note:** Datasets are not included in this repository due to size. Download them separately and place them in your Google Drive as described in the notebook setup cell.

---

## 🏗️ Model Architecture

### U-Net (Segmentation)
```
Input (192×192×1)
  → Encoder: Conv blocks at 32→64→128→256→512 filters with MaxPool
  → Bottleneck: 512 filters, Dropout 0.3
  → Decoder: UpSampling + Skip connections at 256→128→64→32
  → Output: Conv2D(4, 1×1, softmax) → (192×192×4)
```

- **Loss:** 0.5 × Weighted Dice Loss + 0.5 × Weighted Cross-Entropy
- **Optimizer:** Adam (LR=3e-4, clipnorm=1.0)
- **Class weights:** Median-frequency balancing (CSF weighted highest)
- **Augmentation:** Random horizontal/vertical flip + brightness jitter ±0.1

### Tumor Classifiers (BraTS)
| Model | Strategy | Test Accuracy | Macro F1 |
|---|---|---|---|
| DenseNet201 + PCA + SVM | Feature extraction → dimensionality reduction → SVM | 93.2% | 0.931 |
| **VGG16 Fine-tuned ★** | **End-to-end fine-tuning** | **94.3%** | **0.941** |
| EfficientNetB3 | Fine-tuning | 92.3% | 0.920 |

---

## 📈 Key Results

| Metric | Value |
|---|---|
| GM Dice (val) | **0.914** (↑ from baseline 0.869) |
| WM Dice (val) | **0.851** (↑ from baseline 0.836) |
| CSF Dice (val) | **0.861** (↑ from baseline 0.584) |
| Best classifier accuracy | **94.3%** (VGG16) |
| Pipeline end-to-end accuracy | **100%** (4/4 correct) |

### Biomarker Reference Ranges (IBSR Healthy Cohort)

| Biomarker | Mean | Std | Normal Range |
|---|---|---|---|
| GM Volume (cm³) | 237.0 | 43.3 | 150.5 – 323.6 |
| WM Volume (cm³) | 128.7 | 29.5 | 69.8 – 187.7 |
| CSF Volume (cm³) | 6.0 | 2.4 | 1.3 – 10.8 |
| GM/WM Ratio | 1.864 | 0.149 | 1.566 – 2.161 |
| BPF | 0.984 | 0.005 | 0.973 – 0.994 |

---

## 🚀 Running the Project

### 1. Google Colab (Recommended)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Place datasets at:
# /content/drive/MyDrive/Brain_Tissue_Analysis_and_Segmentation/IBSR/
# /content/drive/MyDrive/Brain_Tissue_Analysis_and_Segmentation/BraTS/
```

Then run all cells in the notebook sequentially.

### 2. Running the Streamlit Dashboard (in Colab)

```python
# Install dependencies
!pip install streamlit plotly pillow -q

# Install cloudflared tunnel (no account needed)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
     -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

# Write app.py (paste app.py content or upload to Drive)
# Then launch:
import subprocess, time, requests, re

proc = subprocess.Popen(
    ["python", "-m", "streamlit", "run", "/content/app.py",
     "--server.port=8501", "--server.headless=true",
     "--server.enableCORS=false", "--server.enableXsrfProtection=false",
     "--server.fileWatcherType=none"],
    stdout=open("/content/streamlit.log","w"),
    stderr=open("/content/streamlit_err.log","w"),
)
time.sleep(8)

tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for _ in range(30):
    line = tunnel.stdout.readline().decode()
    m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
    if m:
        print(f"✓ Open: {m.group(0)}")
        break
    time.sleep(1)
```

### 3. Local Machine

```bash
git clone https://github.com/YOUR_USERNAME/Brain_Tissue_Analysis_and_Segmentation.git
cd Brain_Tissue_Analysis_and_Segmentation
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow>=2.12
numpy
pandas
matplotlib
scikit-learn
plotly
streamlit
pillow
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔬 Disease Support Scoring Rubric

The disease support score (0–100) is computed from biomarker Z-score deviations:

| Deviation | Points | Clinical meaning |
|---|---|---|
| BPF > 2σ below normal | +35 | Severe brain atrophy |
| BPF 1–2σ below normal | +15 | Mild atrophy |
| GM/WM > 2σ below normal | +30 | Severe cortical thinning |
| GM/WM 1–2σ below normal | +12 | Borderline cortical thinning |
| CSF > 2σ above normal | +25 | Severe ventricular expansion |
| CSF 1–2σ above normal | +10 | Mild ventricular expansion |
| GM volume > 2σ below normal | +10 | Low gray matter volume |

**Score thresholds:** Normal < 15 · Mild 15–34 · Moderate 35–59 · Notable ≥ 60

Patients scoring ≥ 35 are automatically routed to tumor classification.

---

## 🖥️ Streamlit Dashboard Pages

| Page | Content |
|---|---|
| 📊 Dataset Overview | Dataset cards, voxel class distribution, class weights, pipeline diagram, training config |
| 🔬 Segmentation Results | Per-patient Dice bar chart, validation summary, per-patient radar chart |
| 📈 Volumetric & Biomarker Analysis | Reference ranges table, predicted vs GT comparison, scatter plot, error histogram |
| 🏥 Disease Support | Risk score bar chart, risk distribution pie, patient detail table, scoring rubric |
| 🧬 Tumor Classification | Model accuracy comparison, per-class F1 heatmap, radar chart, full comparison table |
| 🖼️ Live Demo | Upload MRI → segmentation overlay → biomarkers → disease gauge → tumor prediction |

---




| Name | Roll No. | 
|---|---|
| Danushree R S | 23PD05 |
| Harini Sree J | 23PD14 |

---

*Built with TensorFlow · Streamlit · Plotly · scikit-learn*
