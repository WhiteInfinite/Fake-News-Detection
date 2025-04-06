### 1)
# Model Overview
This model is designed to classify news articles as real or fake based on their textual content. It uses a BERT-based transformer model (bert-base-uncased) fine-tuned on a custom dataset of news articles. The model predicts whether a given article is fake or real with high accuracy.


# Datasets Used
The model was trained on a variety of datasets, including:

- Fake News Dataset: Contains labeled news articles with "fake" or "real" classifications.
- News Articles Dataset: A collection of news articles used for training and validation.
# Languages
The model primarily works with English-language news articles, but it could be extended to other languages with appropriate data.

# Metrics
The model's performance was evaluated on the following metrics:

- Accuracy: 99.58%
- Precision: 99.27%
- Recall: 99.88%
- ROC-AUC: 99.99%
- F1-Score: 99.57%
# Model Details
- Base Model: bert-base-uncased
- Fine-Tuning: The model was fine-tuned on a news dataset with labeled examples of real and fake news.
- Training Epochs: 3
- Batch Size: 32
- Optimizer: Adam with weight decay
- Learning Rate: 2e-5



# 


### 2)
Upon comparing the website:
  [https://www.ndtv.com/business-news/union-budget-2025-live-updates-finance-minister-nirmala-sitharaman-economy-finance-income-tax-7602433](https://www.ndtv.com/business-news/union-budget-2025-live-updates-finance-minister-nirmala-sitharaman-economy-finance-income-tax-7602433) 
with 5 other similar news articles, we determined that:
- It is likely to be a legitimate news source.
- The average similarity score obtained with the other 5 articles was around 45%.

# Screenshots


![Screenshot 2025-02-01 210815](https://github.com/user-attachments/assets/eaecdb1f-a02a-4701-bbb0-3626fe0073f1)



![image](https://github.com/user-attachments/assets/e1ec9a9b-7d8b-46ff-ae6e-6302864a5bc0)



![Screenshot 2025-02-01 210832](https://github.com/user-attachments/assets/7eff4b59-2726-4008-a447-d8ce34f6bda0)
