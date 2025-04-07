from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import scipy
import os
from newspaper import Article

app = Flask(__name__)

# Initialize model and load dataset ONCE at startup
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Load and preprocess dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'all-data.csv')

def load_dataset():
    """Load and preprocess the dataset"""
    df = pd.read_csv(DATASET_PATH, 
                   encoding='unicode_escape',
                   names=['Sentiment', 'Text'])
    
    # Add your preprocessing steps here
    df['clean_text'] = df['Text'].str.lower()  # Example preprocessing
    return df

# Load dataset when Flask starts
dataset = load_dataset()

@app.route('/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Endpoint to get dataset information"""
    return jsonify({
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "sentiment_distribution": dataset['Sentiment'].value_counts().to_dict()
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get URL from request
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({"error": "URL is required"}), 400

        # Configure and download article
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}
        
        article = Article(url, browser_user_agent=headers["User-Agent"])
        article.download()
        article.parse()
        
        # Get article title
        title = article.title

        # Analyze sentiment
        with torch.no_grad():
            input_sequence = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=512)
            logits = model(**input_sequence).logits
            scores = {
                k: v for k, v in zip(
                    model.config.id2label.values(),
                    scipy.special.softmax(logits.numpy().squeeze()),
                )
            }
            predicted_sentiment = max(scores, key=scores.get)
            probability = max(scores.values())


        return jsonify({
            "url": url,
            "title": title,
            "sentiment": predicted_sentiment,
            "probability": float(probability),
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)