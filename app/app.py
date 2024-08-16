import os
from flask import Flask, request, Response
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import torch
import requests
import json

app = Flask(__name__)
CORS(app)

# Initialize Pinecone with the correct API key and environment
pc = Pinecone(api_key="0e0c51a7-0388-466b-bb21-1f0a3ab9392f")
index = pc.Index("cve-index-768")

embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")
embedding_model.eval()

def generate_bert_embeddings(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()

# Function to generate answers using your self-hosted model on AWS with chat history
def generate_answer_with_self_hosted_model(query_text, context, history=[]):
    # Set the URL to your self-hosted model API
    url = "http://ollama.ollama.svc.cluster.local:11434/api/chat"
    headers = {
        "Content-Type": "application/json",
    }
    
    # Construct messages with the history included
    messages = [{"role": "user", "content": f"Context: {context}\n\n"}]
    messages.extend(history)  # Include past conversation history
    
    # Add the new query to the messages
    messages.append({"role": "user", "content": f"Question: {query_text}"})
    
    data = {
        "model": "llama3.1:8b",
        "messages": messages,
        "stream": False,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        answer = response.json()["message"]["content"]
        return answer.strip()
    else:
        raise Exception(f"Failed to generate answer: {response.status_code}, {response.text}")

# Function to aggregate the entire metadata from Pinecone results and generate an answer
def generate_answer_for_full_metadata(query_text, pinecone_results, history=[]):
    # Concatenate all metadata fields into a single context string
    full_context = ""
    for match in pinecone_results['matches']:
        metadata = match['metadata']
        metadata_context = " ".join([f"{key}: {value}" for key, value in metadata.items()])
        full_context += metadata_context + " "
    
    # Generate a single answer using the full context
    answer = generate_answer_with_self_hosted_model(query_text, full_context, history)
    
    return {
        'context': full_context,
        'answer': answer
    }

@app.route('/cve-chatbot/query', methods=['POST'])
def query_pinecone():
    data = request.json
    query_text = data.get("query", "")
    top_k = data.get("top_k", 5)
    history = data.get("history", [])

    if not query_text:
        return Response(json.dumps({"error": "Query text is required"}), status=400, mimetype='application/json')

    # Generate the query embedding
    query_embedding = generate_bert_embeddings(query_text)

    # Query the Pinecone index
    pinecone_results = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )

    # Generate answer using the full metadata context from Pinecone results
    result = generate_answer_for_full_metadata(query_text, pinecone_results, history)

    if not result:
        return Response(json.dumps({"error": "No relevant answer found"}), status=404, mimetype='application/json')

    # Update history with the latest question and answer
    history.append({"role": "user", "content": f"Question: {query_text}"})
    history.append({"role": "assistant", "content": result['answer']})

    # Return the answer as JSON
    response = {
        'query': query_text,
        'answer': result['answer']
    }

    return Response(json.dumps(response), status=200, mimetype='application/json')

@app.route('/cve-chatbot')
def home():
    return "<h1>Welcome to the CVE Search Application</h1><p>Use the /query endpoint to search for similar CVEs.</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
