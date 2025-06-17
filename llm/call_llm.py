import os
from dotenv import load_dotenv
from groq import Groq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from qa_chain.llm_template import prompt_template, refine_template

def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None):
    if model in ["llama3-70b-8192"]:
        if api_key == None:
            api_key = os.environ['GROQ_API_KEY']
        llm = Groq(api_key=api_key, model=model , temperature=temperature)
    return llm

template = prompt_template("v2")

def get_completion(prompt:str, model:str, temperature:float=0.1, api_key:str=None):
    if model in ["llama3-70b-8192"]:
        return get_completion_groq(prompt, model, temperature, api_key)
    

def get_completion_groq(prompt:str, model:str, temperature: float, api_key:str):
    if api_key == None:
        api_key = parse_llm_api_key("llama70")
    Groq.api_key = api_key

    messages = [{"role": "user", "content": prompt}]
    response = Groq.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1
    )
    return response.choices[0].message.content.strip()

def parse_llm_api_key(model:str, env_file:dict()=None):
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ

    if model == "Groq":
        return env_file['GROQ_API_KEY']
    else:
        raise ValueError(f"model {model} not support!!")