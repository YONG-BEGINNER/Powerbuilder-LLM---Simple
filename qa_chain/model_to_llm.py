import os
from groq import Groq
from langchain.prompts import PromptTemplate
from llm.call_llm import parse_llm_api_key

def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None):
    if model in ["llama3-70b-8192"]:
        if api_key == None:
            api_key = parse_llm_api_key("groq")
        llm = Groq(api_key=api_key, model=model , temperature=temperature)
    else:
        raise ValueError(f"Model {model} not support!!")
    return llm