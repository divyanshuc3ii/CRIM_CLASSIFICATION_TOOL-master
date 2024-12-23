import os
import torch
import json
import spacy
import nltk
import re
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# import logging
# logger=logging.get_name(__)
# Ensure NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load tokenizer and model
MODEL_NAME = 'l3cube-pune/hing-roberta'
#BEST_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'BEST')
BEST_MODEL_PATH = "/Users/lblb/Desktop/CRIM_CLASSIFICATION_TOOL-master/crime-classifier/backend/ml_model"


LABEL_MAPPING_PATH = "/Users/lblb/Desktop/CRIM_CLASSIFICATION_TOOL-master/crime-classifier/backend/ml_model/label_mapping_WO_CV.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH)
model.eval()

# Load label mapping
with open(LABEL_MAPPING_PATH, "r") as label_file:
    LABEL_MAPPING = json.load(label_file)
    LABEL_MAPPING = {int(k): v for k, v in LABEL_MAPPING.items()}

    
# for k,v in LABEL_MAPPING.items():
#     logger.info(k,v)
# Text Preprocessing Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def mask_sensitive_entities(text):
    doc = nlp(text)
    sensitive_entity_types = {
        "PERSON", "EMAIL", "PHONE", "CARDINAL", 
        "MONEY", "DATE", "ADDRESS"
    }
    
    for ent in doc.ents:
        if ent.label_ in sensitive_entity_types:
            text = text.replace(ent.text, f"<{ent.label_}>")
    
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'because'}
    return ' '.join(word for word in word_tokenize(text) if word.lower() not in stop_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

def predict_crime(texts):
    """
    Perform inference on input texts.
    
    Args:
        texts (list): List of input texts.
        
    Returns:
        list: Predictions and corresponding probabilities.
    """
    # Preprocess texts
    processed_texts = []
    for text in texts:
        cleaned_text = clean_text(text)
        masked_text = mask_sensitive_entities(cleaned_text)
        no_stopwords_text = remove_stopwords(masked_text)
        lemmatized_text = lemmatize_text(no_stopwords_text)
        processed_texts.append(lemmatized_text)
    
    # Tokenize
    inputs = tokenizer(
        processed_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
    
    # Map predictions to labels with probabilities
    results = []
    for pred, prob in zip(predictions, probabilities):
        label = LABEL_MAPPING[pred]
        results.append({
            'label': label,
            # 'probabilities': prob.tolist(),
            'confidence': prob[pred].item()
        })
    
    return results

@csrf_exempt
@require_http_methods(["POST"])
def crime_classification_view(request):
    """
    Django view to handle crime classification requests.
    """
    try:
        # Parse JSON request body
        data = json.loads(request.body)
        text = data.get('text', '')
        
        # Validate input
        if not text:
            return JsonResponse({
                'error': 'No text provided'
            }, status=400)
        
        # Predict (wrapping single text in a list)
        predictions = predict_crime([text])
        
        return JsonResponse({
            'prediction': predictions[0]  # Return first (and only) prediction
        })
    
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

        
      