from fastapi import FastAPI
from pydantic import BaseModel
import os, sys
from qa_chain.QA_chain_self import QA_chain_self
from qa_chain.llm_template import prompt_template

app = FastAPI()

template = prompt_template("v2")

class Item(BaseModel):
    prompt : str
    model : str = "llama3-70b-8192"
    temperature : float = 0.1
    if_history : bool = False
    api_key :str = None
    secret_key : str = None
    db_path : str = "./database/vector_db/chroma"
    file_path : str ="./databse/knowledge_db"
    prompt_template : str = template
    input_variables : list = ["context", "question"]
    embedding : str = "m3e"
    top_k : int = 5
    embedding_key : str = None

@app.post("/")
async def get_response(item:Item):
    if not item.if_history:
        if item.embedding_key == None:
            item.embedding_key = item.api_key
        chain = QA_chain_self(model=item.model, temperature=item.temperature, top_k=item.top_k, file_path=item.file_path,
                              persist_path=item.db_path, embedding=item.embedding, template=template, embedding_key=item.embedding_key)
        
        response = chain.answer(qeustion = item.prompt)

        return response
    
    else:
        return "API Not support history Chatting."