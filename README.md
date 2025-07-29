
# ✈️ Sentiment Analysis for Airline-Based Tweets

A comprehensive deep learning project that analyzes customer sentiments expressed in airline-related tweets using a range of models including LSTM, BiLSTM, BERT, RoBERTa, and an ensemble of these, with LIME explainability support.

---

### 📋 Table of Contents

- [✈️ Project Title](#️-project-title)
- [✅ Features](#-features)
- [📁 Folder & File Structure](#-folder--file-structure)
- [⚙️ Installation](#-installation)
- [▶️ Usage / How to Run](#️-usage--how-to-run)
- [💻 Technologies Used](#-technologies-used)
- [📊 Screenshots / Output Samples](#-screenshots--output-samples)
- [🙏 Credits / Acknowledgements](#-credits--acknowledgements)
- [📄 License](#-license)

---

## ✅ Features

- Cleaned and preprocessed real-world airline tweets dataset
- Multiple deep learning models:
  - LSTM, BiLSTM (with and without GloVe)
  - Transformer-based: BERT, RoBERTa
  - BERTweet for emoji-aware sentiment analysis
- Ensemble model with stacked generalization using XGBoost
- Model explainability using LIME
- Cross-validation for robust evaluation
- Modular code and saved models for easy deployment

---

## 📁 Folder & File Structure

```
Sentimental-Analysis-main/
  .gitignore
  01_EDA_Cleaning.ipynb                # Performs EDA and tweet preprocessing
  02_LSTM_Model.ipynb                  # LSTM-based sentiment classifier
  03_BiLSTM_Model.ipynb                # BiLSTM-based classifier
  04_BiLSTM_GloVe_Model.ipynb          # BiLSTM with GloVe embeddings
  05_BERT_Model.ipynb                  # Sentiment classification using BERT
  BERTweet_emoji_cleaned.ipynb         # BERTweet model with emoji handling
  Cross_Validation_cleaned.ipynb       # K-Fold cross-validation evaluation
  Ensemble_LIME (1)_cleaned.ipynb      # Duplicate or cleaned ensemble version
  Ensemble_LIME.ipynb                  # LSTM + BERT + RoBERTa + XGBoost + LIME
  LSTM_BERT_ROBERTA_XGboostmodel.ipynb # Main ensemble training and prediction
    data/Tweets.csv                    # Original tweet sentiment dataset
  environment.yml                      # Conda environment for reproducibility
  label_encoder.pkl                    # Encoded labels (0, 1, 2)
  sentiment_bilstm_model.keras         # Saved BiLSTM model
  sentiment_lstm_model.keras           # Saved LSTM model
  shared_preprocessing.py              # Text preprocessing utilities
  tokenizer.pkl                        # Tokenizer used in LSTM-based models
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/MasaniselviGanesan/Sentimental-Analysis.git
cd Sentimental-Analysis

# Create conda environment from provided YAML file
conda env create -f environment.yml
conda activate sentiment-analysis
```

> If not using Conda, install manually:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage / How to Run

1. **Start with EDA**:
   - `01_EDA_Cleaning.ipynb`

2. **Train base models**:
   - `02_LSTM_Model.ipynb`
   - `03_BiLSTM_Model.ipynb`
   - `04_BiLSTM_GloVe_Model.ipynb`

3. **Use transformers**:
   - `05_BERT_Model.ipynb`
   - `BERTweet_emoji_cleaned.ipynb`

4. **Evaluate with Cross-Validation**:
   - `Cross_Validation_cleaned.ipynb`

5. **Run full ensemble + explainability**:
   - `LSTM_BERT_ROBERTA_XGboostmodel.ipynb`
   - `Ensemble_LIME.ipynb`

---

## 💻 Technologies Used

- Python
- TensorFlow / Keras
- PyTorch (via HuggingFace Transformers)
- Scikit-learn
- XGBoost
- LIME (Local Interpretable Model-Agnostic Explanations)
- Pandas, NumPy, Matplotlib
- Jupyter Notebook

---

## 📊 Screenshots / Output Samples

📊 **Ensemble Classification Report**:

```
              precision    recall  f1-score   support

    negative       0.99      0.91      0.95      1835
     neutral       0.81      0.94      0.87       620
    positive       0.89      0.98      0.93       473

    accuracy                           0.93      2928
```

🧠 **LIME Explanation** (from notebook):
- Highlights which words in a tweet contributed most to the predicted sentiment class.

---

## 🙏 Credits / Acknowledgements

- Dataset: [Kaggle Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Hugging Face Transformers (BERT, RoBERTa)
- VinAI Research (BERTweet)
- Ribeiro et al. – LIME
- Internship at NIT Trichy

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
