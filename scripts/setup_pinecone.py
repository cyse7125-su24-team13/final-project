import pinecone

# Initialize Pinecone with the correct API key and no environment
pinecone.init(api_key='YOUR_API_KEY', environment=None)

# Connect to the Pinecone index using the specified URL
index = pinecone.Index("cve-index", index_url="https://cve-index-nmt3nw0.svc.aped-4627-b74a.pinecone.io")

# Describe the index to ensure connection is successful
index_stats = index.describe_index_stats()
print(index_stats)
