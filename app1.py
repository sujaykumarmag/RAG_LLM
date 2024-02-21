



import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import os





# Helper functions
def get_embeddings(data):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding =  model.encode(data)
    return embedding

def generate_context(prompt,n_results=1):
    embedding = get_embeddings(prompt).tolist()
    results = collection.query(query_embeddings = embedding, n_results=n_results)
    string = "\n".join(str(item) for item in results["documents"][0])
    return string


def chat_completion(system_prompt, user_prompt,length=1000):
    final_prompt=f"""<s>[INST]<<SYS>>{system_prompt}<</SYS>>{user_prompt} [/INST]"""
    return client.text_generation(prompt=final_prompt)
import os
import chromadb

DATA_DIR = 'data/pythermalcomfort/'


def get_file_paths(directory):
    file_paths = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            file_paths.extend(get_file_paths(item_path))
    return file_paths



def get_file_contents(file_paths):
    file_contents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content_bytes = file.read()
                content_str = content_bytes
                file_contents.append(content_str)
        except Exception as e:
            strs = str(e)+" Sujay"
            print(strs)

    return file_contents


all_file_paths = get_file_paths(DATA_DIR)
text = get_file_contents(all_file_paths)


chromaclient = chromadb.Client()
collection = chromaclient.get_or_create_collection("rag_llm_p")


ids= [str(x) for x in range(len(text))]
collection.add(documents=text,ids=ids)


client = InferenceClient()



# LLama 2 Model

# Vulture GPU Stack (private server Address)
query = "Currently the output of several models are Python dictionaries, they have several limitations and we should use either: dataclasess NamedTuple from the typing library docs"
context = generate_context(query)

print(context+"\n\n\n")

system_prompt = """You are a helpful AI assistant that can answer questions on Python package (pythermal comfort). Answer based on the context provided. If you cannot find the correct answerm, say I don't know. Be concise and just include the response."""

user_prompt=f"""Based on the context:\n {context} \n Answer the below query: \n {query} """

res = chat_completion(system_prompt,user_prompt)
print(res)
