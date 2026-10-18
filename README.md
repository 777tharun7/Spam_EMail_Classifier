<div align="center">

# Spam Email Classifier

**TF–IDF text features with a Multinomial Naive Bayes baseline**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

## Overview

A notebook-based text classification project that vectorizes email/message text with `TfidfVectorizer` and fits a `MultinomialNB` classifier to distinguish **Ham** (legitimate) from **Spam**.

## Implementation

- Notebook: [Spam_Mail/Spam_Mail_Classifier.ipynb](Spam_Mail/Spam_Mail_Classifier.ipynb)
- Saved estimator artifact: [Spam_Mail/Spam_mail_Classifier.joblib](Spam_Mail/Spam_Mail_Classifier.joblib)
- Features visible in the notebook include text vectorization, a train/test split, and classification metrics.

## Run the notebook

Dependencies visible in the notebook include pandas and scikit-learn; the saved model is loaded with joblib.

```bash
python -m pip install pandas scikit-learn joblib
```

Open the notebook in Jupyter or Google Colab and follow its cells in order. The input dataset/source is not currently documented as a reproducible file in this repository, so confirm the data-loading steps before expecting a fresh environment to run end-to-end.

## Results and validation

The saved notebook output reports **100% accuracy** and precision/recall/F1 of 1.00 on a 2,000-row test set. This is a notebook-reported result, not an independently reproduced benchmark. Before interpreting it as model quality, document the dataset source and split procedure, check for duplicate or overlapping messages, fit preprocessing on training data only, and evaluate against a fresh held-out dataset.

For a spam filter, accuracy alone is insufficient: report per-class precision, recall, F1, confusion matrix, false-positive rate for legitimate messages, and false-negative rate for spam.

## Limitations and privacy

Spam patterns change over time. Avoid committing private email content, addresses, personal data, credentials, or API keys. A high score on one test split does not guarantee future performance.

## Repository

[Open source repository](https://github.com/777tharun7/Spam_EMail_Classifier)
