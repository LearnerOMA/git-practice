# Here we use sentence trasformer to genrate embeddings

import chromadb
from sentence_transformers import SentenceTransformer
from scrapData import scrapdata


# Corrected Embedding Function
class SentenceTransformerEmbeddingFunction:
    """Custom Embedding Function using Sentence Transformers."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        # The new version expects 'input' instead of 'texts'
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()



# Load Sentence Transformer model
sentence_transformer_embedding_function = SentenceTransformerEmbeddingFunction()

# Create ChromaDB client with updated embeddings
client = chromadb.PersistentClient(path="./chroma_db")  # Saves data in "chroma_db" folder

# Create collection with new embeddings
collection = client.get_or_create_collection(
    name="NEWS",
    embedding_function=sentence_transformer_embedding_function,
    metadata={"distance_function": "cosine"}  # Ensures cosine distance is used
)

# Fetch collection
def get_collection():
    return collection

# Scrape and add data to collection
def get_data(query_text):
    print("----" * 5, "query function", "----" * 5)
    dataset = scrapdata(scrapTopic=query_text)
    
    for data in dataset:
        print("----" * 20)
        entry = collection.get(ids=[data["Headline"]])
        
        # Check if entry exists before adding
        if not bool(entry["documents"]):
            print("Adding new data to collection")
            collection.add(
                ids=[data["Headline"]],
                documents=[data["Full_Article"]],
                metadatas=[{"source": data["Link"]}]
            )
            print("Added data to collection:", data["Headline"])


# Query collection for similar documents
def ask_question(question):
    print("----" * 5, "ask question def", "----" * 5)
    result = collection.query(
        query_texts=[question],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    return result


# Get answer based on query
def get_answer(question):
    print("----" * 5, "get_answer function", "----" * 5)
    result = ask_question(question)
    
    # Check distance with threshold (cosine distance should be < 0.5)
    if bool(result["documents"][0]) and result["distances"][0][0] > 1.3:
        print("No relevant data found")
        print("Scraping data .......")
        get_data(question)
        result = ask_question(question)
    
    if not result["documents"][0]:
        print("No relevant data found after scraping.")
        return "No relevant data found"
    
    print("Relevant data found")
    return result
