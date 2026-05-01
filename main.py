import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Download stopwords
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenizing and removing stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def main():
    print("--- VerityAI: Fake News Detection System (Python/CLI) ---")
    
    # Load dataset (Creating a dummy one for demonstration, replace with real CSV)
    data = {
        'text': [
            "Scientific study shows that drinking water is healthy for humans.",
            "NASA confirms new planet discovered in Andromeda galaxy.",
            "Alien lizards have infiltrated the Senate and are eating all the chips.",
            "Secret study reveals that the moon is actually made of blue cheese.",
            "World leaders meet to discuss climate change initiatives.",
            "Eating spoons every morning will make you live for 500 years.",
            "The stock market went up by 2% today after economic report.",
            "Giant flying squirrels have taken over the city hall of Paris.",
            "Health experts recommend 8 hours of sleep for optimal function.",
            "Drinking gold liquid will give you the ability to fly through walls."
        ],
        'label': ['REAL', 'REAL', 'FAKE', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE']
    }
    df = pd.DataFrame(data)
    
    # Preprocessing
    print("Preprocessing text data...")
    df['text'] = df['text'].apply(preprocess_text)
    
    # Feature Extraction (TF-IDF)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='REAL'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, pos_label='REAL'):.2f}")
    
    # Real-time Prediction
    print("\n--- Real-time Prediction ---")
    while True:
        user_input = input("\nEnter news text to analyze (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input])
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector).max()
        
        print(f"\nPrediction: {prediction}")
        print(f"Confidence Score: {probability:.2f}")

if __name__ == "__main__":
    main()
