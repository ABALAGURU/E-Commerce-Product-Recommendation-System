from flask import Flask, jsonify, request, render_template
import pandas as pd
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Define the file paths
csv_file_path = 'data/preprocessed.csv'
search_history_file = 'data/search_history.json'

# Load product data
df = pd.read_csv(csv_file_path)

# Initialize Tfid
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['product_name']) 

# cosine similarity 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Load search history
def load_search_history():
    if os.path.exists(search_history_file):
        with open(search_history_file, 'r') as f:
            return json.load(f)
    return []

# Save updated search history
def save_search_history(history):
    with open(search_history_file, 'w') as f:
        json.dump(history, f)

# Recommendation function 
def get_recommendations(query):
    
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    
    top_indices = similarity_scores.argsort()[:-6:-1]  
    recommendations = df.iloc[top_indices]['product_name'].tolist()
    
    if recommendations:
        return recommendations
    else:
        return ["No matching products found."]

# html page
@app.route('/')
def home():
    return render_template('index.html')

# handle search and recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    search_query = data.get('query', '').strip()

    if search_query:
        search_history = load_search_history()
        search_history.append(search_query)
        save_search_history(search_history)

        recommendations = get_recommendations(search_query)
        return jsonify({"recommendations": recommendations, "history": search_history}), 200
    else:
        return jsonify({"error": "No search query provided"}), 400

# route for search history
@app.route('/search_history', methods=['GET'])
def get_search_history():
    history = load_search_history()
    return jsonify({"search_history": history}), 200

if __name__ == '__main__':
    app.run(debug=True)
