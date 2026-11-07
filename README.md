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

The notebook generates a **synthetic 10,000-row dataset** by repeatedly sampling from only 10 spam phrase templates and 10 ham phrase templates (5,000 rows per class). The saved notebook reports **100% accuracy** and precision/recall/F1 of 1.00 on a 2,000-row random test split. Because identical template messages can appear in both training and test splits, this result is vulnerable to duplicate/template leakage and measures recognition of this small synthetic phrase set—not generalisation to real email. It has not been independently rerun during this audit.

Do not present 100% as a general spam-detection accuracy claim. For a defensible evaluation, use a documented real-world or appropriately licensed benchmark, deduplicate before splitting, use a stratified held-out test set (preferably also a time-based split), and report per-class precision, recall, F1, confusion matrix, false-positive rate for legitimate messages, and false-negative rate for spam.

## Limitations and privacy

Spam patterns change over time. Avoid committing private email content, addresses, personal data, credentials, or API keys. A high score on one test split does not guarantee future performance.

## Repository

[Open source repository](https://github.com/777tharun7/Spam_EMail_Classifier)
