# Require Pacakge
# pip install rapidocr_onnxruntime 
# pip install "unstructured[all-docs]" 
# pip install pyMuPDF

# To apply Embedding
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def get_embedding(embedding: str, embedding_key:str=None, env_file:str = None):
   if embedding == "m3e":
      return HuggingFaceEmbeddings(model = "moka-ai/m3e-base")#, model_kwargs = {'device': device})