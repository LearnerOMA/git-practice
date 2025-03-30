import chromadb
import ollama
from scrapData import scrapdata

from chromadb.utils import embedding_functions
class MistralEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom Embedding Function using Mistral via Ollama."""
    
    def __call__(self, texts):
        embeddings = []
        for text in texts:
            response = ollama.embeddings("mistral", text)
            embeddings.append(response['embedding'])
        return embeddings
    

mistral_embedding_function = MistralEmbeddingFunction()
client = chromadb.PersistentClient(path="./chroma_db")  # Saves data in "chroma_db" folder
collection = client.get_or_create_collection(name="NEWS", embedding_function=mistral_embedding_function, metadata={"distance_function": "cosine"})
def get_collection():
    return collection
def get_data(query_text):
    print("----"*5 , "query function" , "----"*5)
    dataset = scrapdata(scrapTopic=query_text)
    for data in dataset:
        # print("Topic:", data["Topic"])
        # print("Headline:", data["Headline"])
        # print("Link:", data["Link"])
        # print("Date:", data["Date"])
        # print("Full Article:", data["Full_Article"])
        # print("Article Summary:", data["Article_Summary"])

        print("----" * 20)
        entry = collection.get(ids=[data["Headline"]])
        print("Entry:", entry)
        if(not bool(entry["documents"])):
            print("Adding new data to collection")
            collection.add(ids=[data["Headline"]], documents=[data["Full_Article"]], metadatas=[{"source": data["Link"]}])
            print("Added data to collection", data["Headline"])
def ask_question(question):
    print("----"*5 , "ask question def" , "----"*5)
    result = collection.query(query_texts=[question], n_results=3, include=["documents", "metadatas", "distances"])
    return result
def get_answer(question):
    print("----"*5 , "get_answer function" , "----"*5)
    result = ask_question(question)
    if(bool(result["documents"][0]) and result["distances"][0][0] > 0.5):
        print("No relevant data found")
        print("Scraping data .......")
        get_data(question)
        result = ask_question(question)
        # if(result["distances"][0][0] > 0.5):
        #     print("No relevant data found")
        #     return "No relevant data found"
    else:
        print("Relevant data found")
    return result


# ask_question("my name is rahul")