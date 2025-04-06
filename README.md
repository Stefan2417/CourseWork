# 🗣️ The Impact of Language on Automatic Speaker Verification Systems

> Research project exploring the impact of language and multilingual pretraining on speaker verification systems.

[![License](https://img.shields.io/github/license/Stefan2417/CourseWork)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Models](https://img.shields.io/badge/models-HuggingFace-orange)](https://huggingface.co/Sttefan/speaker-verification-checkpoints)

---

## 📖 Overview

This repository contains the implementation and experimental setup of a research project on **speaker verification (SV)** in **multilingual and low-resource conditions**. The study evaluates how **multilingual pretraining** affects the **robustness** and **generalization** of deep SV models like `Wav2Vec2-BERT 2.0`, `XEUS`.

---

### 💡 Key Contributions

- ✅ Demonstrated the **importance of multilingual pretraining** for speaker verification tasks, especially under data scarcity and language mismatch.
- 🔧 Successfully **adapted multilingual SSL models** (XEUS, Wav2Vec2-BERT 2.0) to the speaker verification task using the **PMFA method**.

---

## 🧪 Experiments

### ✅ Models Used
- **[Wav2Vec2-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)** (SSL + BERT-style Transformer)
- **[XEUS](https://huggingface.co/espnet/xeus)** (E-Branchformer-based, multilingual SSL)
- **ECAPA-TDNN** (Strong supervised baseline)

### 🌍 Evaluation Datasets
| Dataset           | Type           | Description                         |
|------------------|----------------|-------------------------------------|
| VoxCeleb1 / 2    | Multilingual        | Standard SV benchmark               |
| SL-Celeb (Tamil) | Low-resource   | South Asian speaker verification    |
| SL-Celeb (Sinhala) | Low-resource |                                      |
| isiZulu          | Custom split   | Click language, Southern Africa     |

---

## 🧠 Methodology

Our approach builds upon **PMFA**, where hidden representations from selected layers of pretrained models are concatenated and aggregated using **attentive statistics pooling**, followed by **AAM-Softmax** classification.

📌 See diagram in the paper for detailed design.

---

## 📊 Results

We observed that **multilingual pretrained models significantly outperform monolingual baselines**, especially under data scarcity and language mismatch:

| Model             | VoxCeleb1-O | SL-Celeb-Tamil | Zulu   | VoxSRC21-Val |
|------------------|-------------|----------------|--------|---------------|
| ECAPA-TDNN       | 1.42%       | 3.27%          | 3.30%  | 5.05%         |
| XEUS + PMFA      | 1.29%       | 6.01%          | 2.50%  | 4.06%         |
| W2V2-BERT + PMFA | **0.46%**   | **1.39%**      | **1.90%** | **1.82%**     |

> 📉 Metric: Equal Error Rate (EER)
---

## 🚀 Quickstart

### Setup

1. **Clone the repository and install dependencies:**

    ```bash
    git clone https://github.com/Stefan2417/CourseWork.git
    cd CourseWork
    git lfs install
    git clone https://huggingface.co/espnet/XEUS
    pip install -r requirements.txt
    ./scripts/download_test_pairs_voxceleb.sh
    ```

2. **Download required datasets:**

The project uses the following datasets:

- [VoxCeleb1](https://huggingface.co/datasets/ProgramComputer/voxceleb) and [VoxCeleb2](https://huggingface.co/datasets/ProgramComputer/voxceleb) — required for training and evaluation.
- [NCHLT isiZulu Speech Corpus ](https://repo.sadilar.org/handle/20.500.12185/275) — required for evaluation.
- [SLCeleb](https://ieee-dataport.org/documents/slceleb-speaker-verification) — required for evaluation.
  You need to request access through the official website.

- [MUSAN](https://www.openslr.org/17) - used for noise augmentation.

- [RIRS_NOISES](https://www.openslr.org/28) - used for reverberation.

> Make sure to update all paths in your config files accordingly (see step 3).
> 
3. **Update paths in config files:**

    All configuration files are located in the `src/configs` directory. Make sure to update all absolute paths to match your local environment. These include:

    - Paths to datasets (e.g., MUSAN, RIRS, VoxCeleb)
    - Trial or pair list files
    - Audio directories
    - Pretrained model checkpoints
    - Output directories for logs and model checkpoints

4. **Train the model:**

    ```bash
    python train.py --config src/configs/your_config.yaml
    ```

    > Replace `your_config.yaml` with the specific config file you want to use.

5. **Run evaluation:**

    ```bash
    python inference.py --config src/configs/your_config.yaml
    ```

---


## 📎 Resources

- [Full Report (PDF)](./CourseWork.pdf)
- [Wav2Vec2-BERT Model](https://huggingface.co/facebook/w2v-bert-2.0)
- [XEUS Architecture (ESPnet)](https://huggingface.co/espnet/xeus)

---

## 📦 Checkpoints


Pretrained and fine-tuned model checkpoints used in this project are available on Hugging Face:

👉 [Checkpoints](https://huggingface.co/Sttefan/speaker-verification-checkpoints)

You can use these checkpoints for evaluation or further fine-tuning.

---

## 🙌 Acknowledgements

- Supervised by **Petr Markovich Grinberg**, HSE.
- Experiments conducted using **HSE supercomputing cluster**.
- Thanks to the creators of **ESPnet**, **HuggingFace**, **Transformers**, and **SpeechBrain**.

## 🧾 Credits

Project template are based on  
👉 [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template)
