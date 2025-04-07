from flask import Flask, request, jsonify
import newspaper
from newspaper import Article
from newspaper.configuration import Configuration
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Initialize model once at startup
tokenizer = AutoTokenizer.from_pretrained("Pulk17/Fake-News-Detection")
model = AutoModelForSequenceClassification.from_pretrained(
    "Pulk17/Fake-News-Detection", torch_dtype=torch.float16
)

@app.route('/detect_fake_news', methods=['POST'])
def detect_fake_news():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Invalid request. 'url' key is required in JSON payload"}), 400
        
        url = data.get('url')
        if not url.strip():
            return jsonify({"error": "URL cannot be empty"}), 400

        # Configure article download
        config = Configuration()
        config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0.0.0 Safari/537.36"
        )
        article = Article(url, config=config)
        try:
            article.download()
            article.parse()
        except Exception as e:
            return jsonify({"error": "Failed to process the article", "details": str(e)}), 400
        article.download()
        article.parse()

        # Prepare response data
        article_data = {
            "title": article.title,
            "url": url,
            "content": article.text[:500] + "..." if len(article.text) > 500 else article.text,
            "word_count": len(article.text.split())
        }

        # Fake news detection
        inputs = tokenizer(article.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        labels = ["fake", "real"]
        prediction = labels[predictions.item()]
        
        # Get confidence scores
        probabilities = torch.softmax(logits, dim=-1)
        confidence = round(probabilities[0][predictions.item()].item(), 4)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "article": article_data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "error": "Failed to analyze article"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)