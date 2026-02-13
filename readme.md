# Uncertainty-aware Generative Recommendation

This repository contains the PyTorch implementation for the paper **"Uncertainty-aware Generative Recommendation"**.

## Data Preparation

Please follow the steps below to prepare the datasets and pre-trained components.

### 1. Download Raw Datasets
Download the raw datasets (e.g., Amazon18) from the [Amazon Datasets](https://nijianmo.github.io/amazon/index.html) repository.

### 2. Data Preprocessing
Filter and split the raw data into training, validation, and test sets:
```bash
bash ./data/_1/amazon18_data_process.sh
```

### 3. Text Embedding Generation
Convert textual data into vector representations:
```bash
bash ./data/_2/amazon_text2emb.sh
```

### 4. SIDs Generation
Train the RQ-VAE model:
```bash
bash ./data/_3/rqvae.sh
```
Generate item SIDs:
```bash
python ./data/_3/generate_indices.py
```

## Model Training and Evaluation
### 1. Supervised Fine-Tuning (SFT)
```bash
bash ./train/SFT/run_train_SFT.sh
```

### 2. Reinforcement Learning (RL)
Align the model using the uncertainty-aware reinforcement learning framework:
```bash
bash ./train/RL/run_train_RL.sh
```

### 3. Evaluation
Evaluate the model performance on the test set:
```bash
bash ./eval/run_eval.sh
```

