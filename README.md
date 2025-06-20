# 📚 DLA-Labs — Deep Learning Applications Exam

This repository contains the implementation of three lab projects for the **Deep Learning Applications** exam.
## 📁 Repository Structure

```
DLA-Labs/
├── LAB1/ # ResNet, MLP, CNN implementations
│ └── Lab1-CNN.ipynb
| └──
| └──
├── LAB2/ # Deep Reinforcement Learning experiments
│ └── Lab2-DRL.ipynb
| └── Lunar_best_model_NOValueNet.pth
| └── Lunar_best_model_ValueNet.pth
| └── best_model.pth
├── LAB3/ # Transformers with Hugging Face
│ └── Lab3-transformers.ipynb
```

> 🔍 **Note:** Each folder contains a Jupyter Notebook and optionally additional scripts or resources used for that lab.

---

## 🚀 Project Summaries

### 🧠 LAB 1 - ResNet, MLP, and CNN Architectures
- **Platform:** Google Colab (NVIDIA T4 GPU)
- **Description:** Implementation of standard deep learning models such as Multilayer Perceptrons, Convolutional Neural Networks, and ResNet.
- **Goal:** Compare model performance on a typical image classification task.

### 🕹️ LAB 2 - Deep Reinforcement Learning Laboratory
- **Platform:** Local Jupyter Notebook (NVIDIA GeForce 940MX) on VS Code
- **Description:** Exploration of Deep Reinforcement Learning using OpenAI Gym environments.
- **Algorithms:** DQN, A2C, etc.

### 🤖 LAB 3 - Working with Transformers in the Hugging Face Ecosystem
- **Platform:** Local Jupyter Notebook (NVIDIA GeForce 940MX) on VS Code
- **Description:** Utilizing Hugging Face `transformers` for NLP tasks such as text classification and question answering.
- **Tools:** `transformers`, `datasets`, `tokenizers`

---

## 🔁 Reproducibility Instructions

Each lab is self-contained and can be run directly in Jupyter Notebook using the instructions below. For LAB2 and LAB3, is it necessary to use **conda environments** to ensure proper dependency isolation and GPU compatibility.

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/DLA-Labs.git
cd DLA-Labs
```
### ⚙️ Step 2: Create and Activate Conda Environments
📦 LAB 2 — Deep Reinforcement Learning

Create the environment and activate it:
```
conda create -n DRL -c conda-forge gymnasium pytorch-gpu matplotlib pygame jupyterlab
conda activate DRL
```
Launch Jupyter:
```
jupyter lab
```
Open LAB2/Lab2-DRL.ipynb to run the experiments.
Pre-trained models (*.pth) are already included in the folder for reproducibility.
📦 LAB 3 — Transformers with Hugging Face

Create and activate the environment:
```
conda create -n transformers -c conda-forge transformers datasets matplotlib scikit-learn torchvision pytorch-gpu accelerate sentencepiece jupyterlab ipywidgets tqdm
conda activate transformers
```
Launch Jupyter:
```
jupyter lab
```
Open LAB3/Lab3-transformers.ipynb to run the notebook.
All results are reproducible using the uploaded code and models.

    📝 Note for LAB 1:
    LAB1 was executed in Google Colab using a free NVIDIA T4 GPU. You can run Lab1-CNN.ipynb directly in Colab. No setup is required beyond installing packages within the notebook.


