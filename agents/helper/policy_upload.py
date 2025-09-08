from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Azure Search
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("INDEX_NAME") or "default-index-name"  # Replace "default-index-name" with a valid default value
search_api_key = os.getenv("AZURE_SEARCH_KEY")

# Azure OpenAI
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_ENGINE")  # e.g. "text-embedding-3-small"
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize clients
if index_name is None:
    raise ValueError("INDEX_NAME environment variable is not set.")
if search_service_endpoint is None:
    raise ValueError("AZURE_SEARCH_ENDPOINT environment variable is not set.")
if search_api_key is None:
    raise ValueError("AZURE_SEARCH_KEY environment variable is not set.")
if openai_endpoint is None:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
if openai_api_key is None:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
if embedding_model is None:
    raise ValueError("AZURE_OPENAI_EMBEDDING_MODEL environment variable is not set.")


search_client = SearchClient(search_service_endpoint, index_name, AzureKeyCredential(search_api_key))
aoai_client = AzureOpenAI(api_key=openai_api_key, api_version="2024-05-01-preview", azure_endpoint=openai_endpoint)

# Your policy KB
# Your policy KB
POLICY_KB = [
    ("Spending thresholds",
     "Orders up to £500 total is auto-approved. "
     "Orders above £5,000 require finance Team approval and cannot be auto-approved."),

    ("Quantity limits",
     "Per order quantity limit is 50 units for any hardware item. Larger requests must be split "
     "or go through financial team approval."),

    ("Restricted products",
     "Restricted: Gaming GPUs, crypto miners, unlicensed software."),

    ("Currency and vendor",
     "All pricing should be in GBP(£)/US Dollar($) for approval thresholds; convert if needed."),

    ("Shipping and tax",
     "Include shipping and tax in total price for approval thresholds. Standard shipping is used ")

]

# Upload documents
docs = []
for i, (title, content) in enumerate(POLICY_KB, start=1):
    # Get embedding from Azure OpenAI
    emb = aoai_client.embeddings.create(model=embedding_model, input=content)
    vector = emb.data[0].embedding

    docs.append({
        "id": str(i),
        "title": title,
        "content": content,
        "contentVector": vector
    })

search_client.upload_documents(documents=docs)
print(f"✅ Uploaded {len(docs)} policy documents with embeddings!")
