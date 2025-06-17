from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

class Chat_QA_chain_self:
    def __init__(self, model:str, temperature:float, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, 
                appid:str=None, api_key:str=None, embedding="m3e", embedding_key:str=None):
        self.model = model
        self.temperature=temperature
        self.top_k = top_k
        self.chat_history = chat_history
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid=appid
        self.api_key = api_key
        self.embedding = embedding
        self.embedding_key = embedding_key

        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)

    def clear_history(self):
        #Clear char history
        return self.chat_history.clear()

    def change_history_lenght(self, history_len:int=1):
        # Get recent chat history
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

    def answer(self, question:str=None, temperature = None, top_k = 4):
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
        llm = model_to_llm(self.model, temperature, self.api_key)

        retriever = self.vectordb.as_retirever(search_type="similarity",
                                                search_kwargs={'k': top_k})
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever = retriever
        )

        result = qa({"question": question, "chat_history": self.chat_history}) # Result will include the question, chat_history and answer
        answer = result['answer']
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question, answer)) # Update chat_history 
        
        return self.chat_history