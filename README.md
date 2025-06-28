  # ✈️ Sentimental Analysis on Airline Tweets

> ⚠️ **Note:** This project is currently in-progress and under active development. Some features and results are still being implemented and finalized.

---

## 📌 Overview

This project applies deep learning models (LSTM, Bi-LSTM) to classify airline-related tweets into **positive**, **neutral**, or **negative** sentiments. It’s being developed in **multiple phases**, with comparisons at each stage.

---

## 🚧 Project Status

| Component                |Status        |
|------------------------  | ------------ |
| ✅ Shared Preprocessing  | Complete    |
| ✅ Base LSTM Model       | Implemented |
| ✅ Bi-LSTM Model         | Implemented |
| 🔄 GloVe Embedding       | Coming soon |
| 🔄 BERT Integration      | Coming soon |
| ✅ EDA & Visuals         | Drafted     |
| ✅ GitHub Sync           | Complete    |

---

## 📂 Project Structure

<details>
<summary>Click to view structure</summary>
Sentimental-Analysis/
├── data/
├── notebooks/ # LSTM, Bi-LSTM, EDA
├── scripts/ # Shared preprocessing, predict.py
├── models/ # Saved models (optional)
├── reports/ # Phase-wise PDFs or summaries
├── requirements.txt
└── README.md

</details>

---

## 🛠 Technologies

- Python, TensorFlow, Keras
- Scikit-learn, Matplotlib, Seaborn
- Jupyter Notebook
- WordCloud

---

## ✅ Completed

- ✅ Shared preprocessing pipeline
- ✅ LSTM and Bi-LSTM models
- ✅ Unified training/evaluation framework
- ✅ Inference with saved tokenizer & encoder
- ✅ Exploratory visuals and word clouds
- ✅Integrate GloVe pre-trained embeddings

---

## 🔄 Upcoming


- 🔄 Apply BERT for contextual classification
- 🔄 Handle class imbalance with weighted loss or augmentation
- 🧾 Compile final comparison and report

# 📦 GloVe Embeddings

Download `glove.6B.100d.txt` from:

https://nlp.stanford.edu/data/glove.6B.zip

Extract it and place the file inside this `/data/` folder.

Note: Not included in repo due to GitHub file size limits.


 

## 📌 Note

Please **don’t judge the performance or code quality yet** — this repository is part of an academic deep learning NLP project and still evolving across multiple enhancement phases.


