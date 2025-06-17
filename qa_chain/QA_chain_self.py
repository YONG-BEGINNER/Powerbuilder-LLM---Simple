from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
from qa_chain.llm_template import prompt_template
import re

class QA_chain_self():
    default_template = prompt_template("v1")

    def __init__(self, model:str, temperature:float=0.0, top_k:str=None, file_path:str=None, persist_path:str=None, api_key:str=None, embedding="m3e",
                 embedding_key=None, template=default_template):
        
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.api_key = api_key
        self.embedding  = embedding
        self.embedding_key = embedding_key
        self.template = template
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.api_key)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                              template=self.template)
        self.retriever = self.vectordb.as_retriever(search_type = "similarity",
                                                    search_kwrags ={'k': self.top_k})
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                    retrieval = self.retriever,
                                                    return_source_document=True,
                                                    chain_type_kwargs = {"prompt":self.QA_CHAIN_PROMPT})
        
    def answer(self, question:str=None, temperature=None, top_k=4):
        if len(question) == 0:
            return""
        
        if temperature == None:
            temperature = self.temperature

        if top_k == None:
            top_k = self.top_k

        result = self.qa_chain({"query":question, "temperature":temperature, "top_k":top_k})
        answer = result["result"]
        answer = re.sub(r"\\n",'<br/>', answer)
        return answer
        
