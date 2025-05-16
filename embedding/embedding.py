# Require Pacakge
# pip install rapidocr_onnxruntime 
# pip install "unstructured[all-docs]" 
# pip install pyMuPDF

import os
import time
from langchain_chroma import Chroma
# To read PDF
from langchain_community.document_loaders import PyMuPDFLoader as pymu
# To split the text
from langchain.text_splitter import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# To apply Embedding
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_SIZE = 500
OVERLAP_SIZE = 50
pages = []

start_time = time.time()
for file in os.listdir('./data'):
  if file.endswith('.pdf'):
    loader = pymu(os.path.join('./data', file))
    pages.extend(loader.load())

time_used = time.time() - start_time
print(f"Time Used: {time_used} Second")
print(f"Time Used(Minutes): {time_used/60:.2f}")
print(f"Total Quantity: {len(pages)}")
print(f"Word Quantity(Used to estimate the quantity of token):{sum([len(doc.page_content) for doc in pages])}")

# #Check the data of the page
# print(f"Loaded into type: {type(pages)}\nTotal Page: {len(pages)}")

# #Print out the content of the page
page = pages[33]
# print(f"Element type of the page: {type(page)}\nMetadata of the page: {page.metadata}\nPage Content:\n{page.page_content[0:1000]}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = OVERLAP_SIZE
)

split_docs = text_splitter.split_documents(pages)
print(f"Splited quantity: {len(split_docs)}")
print(f"Word Quantity after splited(Used to estimate the quantity of token):{sum([len(doc.page_content) for doc in split_docs])}")

embedding = HuggingFaceEmbeddings(
    model = "moka-ai/m3e-base"
    # model_kwargs = {'device': device}
)

persist_directory = './database/vector_db/chroma'

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory,exist_ok=True)

# # Apply Embedding to the document
# vectordb = Chroma.from_documents(
#     documents = split_docs,
#     embedding=embedding,
#     persist_directory=persist_directory
# )

# Load VectorDB
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(f"Vectors Quantity: {vectordb._collection.count()}")

question = "What is Power Builder?"
sim_docs = vectordb.max_marginal_relevance_search(question,k=4)
print(f"Search Result: {len(sim_docs)}")

for i, sim_docs in enumerate(sim_docs):
    print(f"Searched {i+1} Result: \n{sim_docs.page_content[:200]}", end="\n---------------\n")