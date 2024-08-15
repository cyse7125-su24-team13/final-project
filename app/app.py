from flask import Flask, request, Response
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from pinecone import Pinecone
import torch
import json

app = Flask(__name__)

# Initialize Pinecone with the correct API key and environment
pc = Pinecone(api_key="0e0c51a7-0388-466b-bb21-1f0a3ab9392f")
index = pc.Index("cve-index-768")

# Load the BERT model and tokenizer for embeddings
embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")
embedding_model.eval()

# Load an LLM model for refining the results
llm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
llm_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
llm_model.eval()

# Function to generate embeddings using BERT
def generate_bert_embeddings(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()

def refine_results_with_llm(query_text, pinecone_results):
    # Prepare the LLM input format
    inputs = [f"Query: {query_text} Context: {match['metadata']['description']}" for match in pinecone_results['matches']]
    llm_inputs = llm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    
    # Get relevance scores from LLM
    with torch.no_grad():
        outputs = llm_model(**llm_inputs)
    relevance_scores = outputs.logits.squeeze().tolist()

    # Attach relevance scores to the results
    for i, match in enumerate(pinecone_results['matches']):
        match['llm_score'] = relevance_scores[i]

    # Sort results by LLM scores and select the top 3
    sorted_results = sorted(pinecone_results['matches'], key=lambda x: x['llm_score'], reverse=True)[:3]

    return sorted_results

def convert_to_serializable(results):
    # Convert ScoredVector objects and other non-serializable objects to dictionaries
    serializable_results = []
    for result in results:
        serializable_result = {
            'id': result['id'],
            'cve_metadata': result['metadata']['cve_metadata'],
            'description': result['metadata']['description'],
            # 'llm_score': result.get('llm_score', None)
        }
        serializable_results.append(serializable_result)
    return serializable_results

@app.route('/cve-chatbot/query', methods=['POST'])
def query_pinecone():
    data = request.json
    query_text = data.get("query", "")
    top_k = 10

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

    # Refine the results with LLM and get the top 3 ranked results
    top_3_results = refine_results_with_llm(query_text, pinecone_results)

    # Handle the case where top_3_results is None
    if top_3_results is None:
        print("Warning: top_3_results is None")
        return Response(json.dumps({"error": "Failed to refine results with LLM"}), status=500, mimetype='application/json')

    # Convert the top 3 results to a serializable format
    serializable_results = convert_to_serializable(top_3_results)

    # Return the top 3 ranked results as JSON
    return Response(json.dumps(serializable_results), status=200, mimetype='application/json')

@app.route('/cve-chatbot')
def home():
    return "<h1>Welcome to the CVE Search Application</h1><p>Use the /query endpoint to search for similar CVEs.</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
