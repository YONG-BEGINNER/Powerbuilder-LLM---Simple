import sys, os
import IPython.display
import io
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
import re

_ = load_dotenv(find_dotenv())

LLM_MODEL_DICT = {
    'groq':['llama3-70b-8192']
}

LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = 'llama3-70b-8192'
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "./database/knowledge_db"
DEFAULT_PERSIST_PATH = "./database/vector_db/chroma"
AIGC_AVATAR_PATH = "./figure"
DATAWHALE_AVATAR_PATH = "./figure"
AIGC_LOGO_PATH = "./figure"
DATAWHALE_LOGO_PATH = "./figure"

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform,"")

class Model_center():
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self= {}

    def chat_qa_chain_self_answer(self, question:str, chat_history: list = [])