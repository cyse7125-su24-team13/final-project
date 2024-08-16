from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import torch

# Initialize Pinecone with the correct API key and environment
pc = Pinecone(api_key="0e0c51a7-0388-466b-bb21-1f0a3ab9392f")

# Connect to the existing Pinecone index with dimension 768
index = pc.Index("cve-index-768")

# Load the BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Ensure the model is in evaluation mode
model.eval()

# Function to generate embeddings using BERT
def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()  # Convert to a list of Python floats

# Function to query the Pinecone index
def query_pinecone(query_text, top_k=5):
    # Generate the query embedding
    query_embedding = generate_bert_embeddings(query_text)
    
    # Debugging: Print the query embedding and its length
    print(f"Query Embedding: {query_embedding[:5]}... (showing first 5 values)")
    print(f"Embedding Length: {len(query_embedding)}")
    
    # Query the Pinecone index
    results = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )

    # Print the results
    for match in results['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')}")

if __name__ == "__main__":
    # Example query
    query_text = "CVE-2021-1234"
    print(f"Querying Pinecone for: '{query_text}'")
    query_pinecone(query_text, top_k=2)
