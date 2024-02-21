
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import os 

# Embeddings 
sentences = ["Mango is a Fruit", "Fuits have a King"]

def text_embedding(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences,normalize_embeddings=True)
    return embeddings
    #print(embeddings)

def vec_similarity(vec1, vec2):
    return np.dot(vec1,vec2) / np.linalg.norm(vec1)*np.linalg.norm(vec2)

vecs = text_embedding(sentences)
similarity_index = vec_similarity(vecs[0],vecs[1])

# Vector DB
phrases=[
    "Amanda baked cookies and will bring Jerry some tomorrow.",
    "Olivia and Olivier are voting for liberals in this election.",
    "Sam is confused, because he overheard Rick complaining about him as a roommate. Naomi thinks Sam should talk to Rick. Sam is not sure what to do.",
    "John's cookies were only half-baked but he still carries them for Mary."
]

ids = ["1","2","3","4"]

metadatas=[{"source": "pdf-1"},{"source": "doc-1"},{"source": "pdf-2"},{"source": "txt-1"}]

chromaClient = chromadb.Client()
collection = chromaClient.create_collection(name="embeddings_demo")
collection.add(documents=phrases,metadatas=metadatas, ids=ids)


query_text = "Mary Got half baked cake"

results = collection.query(
    query_texts=[query_text],
    n_results=2

)

# print(results['documents'][0])

import fitz  # PyMuPDF

# Open the PDF file
pdf_file_path = './data/annualreport.pdf'
pdf_document = fitz.open(pdf_file_path)

# Initialize an empty string to store the extracted text
text = []

# Iterate through each page of the PDF
for page_num in range(pdf_document.page_count):
    # Get the page object
    page = pdf_document.load_page(page_num)
    
    # Extract text from the page
    page_text = page.get_text()
    
    # Append the extracted text to the overall text
    text.append(page_text)

# Close the PDF document
pdf_document.close()

client = chromadb.Client()
fin_collection = client.get_or_create_collection("annual_report")

ids= [str(x) for x in range(len(text))]

fin_collection.add(documents=text,ids=ids)


results = fin_collection.query(query_texts=["What is the Racial Equity Fund for India ?"],n_results=1)

print(results["documents"][0][0])


