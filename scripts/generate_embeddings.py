from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import json
import os
import numpy as np
from tqdm import tqdm
import torch

# Initialize Pinecone with the correct API key and environment
pc = Pinecone(api_key="0e0c51a7-0388-466b-bb21-1f0a3ab9392f")

# Create and connect to a new Pinecone index with dimension 768
index = pc.Index("cve-index-768")

# Load the BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Ensure the model is in evaluation mode
model.eval()

# Directory containing the extracted CVE data
cve_dir = "data/cvelist/cvelistV5-main/cves"

# Function to extract CVE data (cve_id and descriptions)
def extract_cve_data(data_dir):
    cve_data = []

    for root, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for subroot, _, subfiles in os.walk(dir_path):
                for file in subfiles:
                    if file.endswith(".json"):
                        file_path = os.path.join(subroot, file)
                        with open(file_path, 'r') as f:
                            try:
                                cve_record = json.load(f)
                                cve_metadata = cve_record.get('cveMetadata', {})
                                cve_id = cve_record.get('cveMetadata', {}).get('cveId', 'N/A')
                                container_cna = cve_record.get('containers', {}).get('cna', {})
                                if 'descriptions' in container_cna:
                                    for desc in container_cna['descriptions']:
                                        cve_data.append({
                                            'cve_id': cve_id,
                                            'description': desc['value'],
                                            'cve_metadata': cve_metadata
                                        })
                            except json.JSONDecodeError as e:
                                print(f"Error reading {file_path}: {e}")
            break

    return cve_data

# Function to generate embeddings using BERT
def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Function to generate and store embeddings in Pinecone
def generate_and_store_embeddings():
    cve_data = extract_cve_data(cve_dir)
    
    print(f"Extracted {len(cve_data)} CVE records.")

    if not cve_data:
        print("No CVE data extracted.")
        return

    descriptions = [record['description'] for record in cve_data]
    embeddings = []

    for desc in tqdm(descriptions, desc="Generating embeddings", unit="embedding"):
        embedding = generate_bert_embeddings(desc)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    print(f"Generated {len(embeddings)} embeddings.")

    if embeddings.size == 0:
        print("No embeddings generated.")
        return
    
    vectors = [
        {
            "id": cve_data[i]['cve_id'],
            "values": embeddings[i].tolist(),
            "metadata": {
                'cve_metadata': json.dumps(cve_data[i]['cve_metadata']),  # Serialize to JSON string
                'description': cve_data[i]['description']
            }
        }
        for i in range(len(embeddings))
    ]

    print(f"Prepared {len(vectors)} vectors for upsert.")

    if not vectors:
        print("No vectors to upsert.")
        return
    
    index.upsert(vectors=vectors, namespace="ns1")
    
    print("Embeddings generated and stored in Pinecone index.")

if __name__ == "__main__":
    generate_and_store_embeddings()
