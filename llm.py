from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

ChatGroq.api_key = os.environ['GROQ_API_KEY']

def get_completion(prompt,
                   model = "llama3-8b-8192"):
    messages = [{"role":"user", "content":prompt}]
    response = ChatGroq.chat.completions.create(
        model = model,
        messages = messages,
        temperature=0
    )

    return response.choices[0].message.content

persist_directory = './database/vector_db/chroma'

embedding = HuggingFaceEmbeddings(
    model = "moka-ai/m3e-base"
    # model_kwargs = {'device': device}
)

# Load VectorDB
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(f"Vectors Quantity: {vectordb._collection.count()}")

llm = ChatGroq(model_name = "llama3-8b-8192", temperature = 0)
answer = llm.invoke("Hi")
print(answer)