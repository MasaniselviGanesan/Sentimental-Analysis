
# Sentiment Analysis of Airline Tweets

This project presents a sentiment classification model based on tweets directed at major U.S. airlines. The primary objective is to identify the sentiment expressed in each tweet as negative, neutral, or positive.

## Project Structure

- `01_EDA_Cleaning.ipynb`: Contains the data loading, exploration, and preprocessing steps.
- `02_LSTM_Model.ipynb`: Implements an LSTM-based deep learning model for sentiment classification.
- `data/Tweets.csv`: Dataset used for training and evaluation.
- `environment.yml`: Lists the required packages and environment specifications.
- `README.md`: Project overview and setup guide.

## Dataset Overview

The dataset includes approximately 14,600 tweets labeled with sentiment categories and various metadata such as tweet source, user timezone, retweet count, and reason for negative sentiment. A number of fields contain missing values and were cleaned as part of preprocessing.

## Model Summary

The sentiment classification model uses a sequential LSTM network built with the Keras API. The model uses tokenized and padded tweet texts as input and outputs a categorical prediction for sentiment class.

## Acknowledgments

Dataset Source: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
