# 💊 Prescription Medicine Name Recognition using Fine-tuned TrOCR

An OCR + Deep Learning pipeline that reads handwritten prescription images and extracts medicine names using a fine-tuned **TrOCR** (Transformer-based OCR) model.

---

## 📌 Project Overview

Handwritten prescriptions are a major source of medical errors in pharmacies. This project automates the reading of prescription medicine names from images using Microsoft's TrOCR model, fine-tuned on a prescription-specific dataset.

**Key capabilities:**
- Reads handwritten medicine names from prescription images
- Fine-tuned on real prescription image data
- Evaluated using Accuracy, CER (Character Error Rate), and WER (Word Error Rate)

---

## 🧠 Model

- **Base Model:** [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten)
- **Architecture:** Vision Encoder (ViT) + Language Decoder (RoBERTa)
- **Task:** Image → Medicine Name (text generation)

---

## 📂 Dataset

- **Source:** [Kaggle — prescription-preprocessed](https://www.kaggle.com/datasets/arkaloveskaggle/prescription-preprocessed)
- **Format:** JPEG/PNG images of handwritten prescription medicine names
- **Labels:** Extracted from filenames using custom parsing logic

---

## ⚙️ Training Setup

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Scheduler | Linear warmup (100 steps) |
| Device | CUDA (GPU) |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/prescription-ocr.git
cd prescription-ocr
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/arkaloveskaggle/prescription-preprocessed) and place in:
```
/kaggle/input/datasets/arkaloveskaggle/prescription-preprocessed/preprocessed/
```

### 4. Run the notebook
Open `medtech.ipynb` in Jupyter or Kaggle and run all cells.

---

## 📦 Requirements

```
torch
transformers
Pillow
jiwer
tqdm
```

Install with:
```bash
pip install torch transformers Pillow jiwer tqdm
```

---

## 📊 Evaluation Metrics

The model is evaluated on the last 100 images of the dataset using:

Accuracy : 72.00%
CER      : 0.3649
WER      : 0.2485

Loss after 10th epoch =1.6678
---

## 🐛 Known Issues & Findings

- **Label parsing bug:** The `PrescriptionDataset` class uses `filename.split('-')[0]` for label extraction, which fails for multi-part drug names like `Benzyl-Penicillin`. A corrected `get_actual_label()` function was written but not yet integrated into the dataset class — this likely contributed to lower accuracy.
- **Small dataset:** Limited training data affects generalization. Data augmentation is recommended.

---

## 🔮 Future Work

- Fix label extraction to use the improved `get_actual_label()` function
- Add data augmentation (`nlpaug`, `TextAttack`)
- Integrate BioBERT NER on top of TrOCR output for structured extraction (drug, dosage, frequency)
- Build a hybrid pipeline: TrOCR (reads) → LLM (structures) → Agent (acts)

---

## 📁 Project Structure

```
prescription-ocr/
├── medtech.ipynb          ← Main training & evaluation notebook
├── README.md              ← This file
├── requirements.txt       ← Dependencies
├── .gitignore             ← Excludes checkpoints and cache
└── docs/
    └── report.pdf         ← Project report (optional)
```

---

## 🙈 .gitignore

```
checkpoint.pth
*.pth
__pycache__/
.ipynb_checkpoints/
```

---

## 📚 References

- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models — Li et al., 2021](https://arxiv.org/abs/2109.10282)
- [HuggingFace TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)
- [Kaggle Dataset](https://www.kaggle.com/datasets/arkaloveskaggle/prescription-preprocessed)

---

Arka Pal
emailtoarka@gmail.com
KIIT university

---

*Built with ❤️ using PyTorch + HuggingFace Transformers*
