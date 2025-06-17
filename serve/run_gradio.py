import sys, os
import IPython.display
import io
import gradio as gr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
EMBEDDING_MODEL_LIST = ['m3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "./database/knowledge_db"
DEFAULT_PERSIST_PATH = "./database/vector_db/chroma"
AIGC_AVATAR_PATH = "./figure/Profile Pic.jpeg"
DATAWHALE_AVATAR_PATH = "./figure/AI Profile.jpeg"
AIGC_LOGO_PATH = "./figure/AI Logo.png"
DATAWHALE_LOGO_PATH = "./figure/AI Profile.jpeg"

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform,"")

class Model_center():
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self= {}

    def chat_qa_chain_self_answer(self, question:str, chat_history: list = [], model: str="llama3-70b-8192", embedding:str = "m3e",
                                  temperature: float = 0.0, top_k: int = 4, history_len: int = 3, file_path: str =DEFAULT_DB_PATH,
                                  persist_path: str = DEFAULT_PERSIST_PATH):
        
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(model=model, temperature=temperature, top_k=top_k, chat_history=chat_history,
                                                                               file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, chat_history
        
    def qa_chain_self_answer(self, question: str, chat_history: list = [], model:str = "llama3-70b-8192", embedding:str = "m3e",
                             temperature: float = 0.0, top_k: int = 4, file_path:str = DEFAULT_DB_PATH, persist_path:str = DEFAULT_PERSIST_PATH):
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(model= model, temperature=temperature, top_k=top_k,
                                                                       file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.qa_chain_self[(model,embedding)]
            chat_history.append((question, chain.answer(question, temperature,top_k)))
            return "", chat_history
        except Exception as e:
            return e, chat_history
    
    def clear_history(self):
        if len(self.chat_qa_chain_self)>0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()

def format_chat_prompt(message, chat_history):
    prompt=""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser:{user_message}\nAssistance:{bot_message}"
    prompt = f"{prompt}\nUser:{message}\nAssistant:"
    return prompt
            
def respond(message, chat_history, llm, history_len=3, temperature=0.1):
    if message == None or len(message) < 1:
        return "", chat_history
    try:
        if chat_history and history_len > 0:
            chat_history = chat_history[-history_len:]
        elif history_len <= 0:
            chat_history = []
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = get_completion(formatted_prompt, llm, temperature=temperature)
        bot_message = re.sub(r"\\n",'<br/>', bot_message)
        chat_history.append((message, bot_message))
        return "", chat_history
    except Exception as e:
        return e, chat_history
    
model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):           
        gr.Image(value=AIGC_LOGO_PATH, scale=0.3, min_width=10, show_label=False, show_download_button=False, container=False)
   
        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>动手学大模型应用开发</center></h1>
                <center>LLM-UNIVERSE</center>
                """)
        gr.Image(value=DATAWHALE_LOGO_PATH, scale=0.3, min_width=10, show_label=False, show_download_button=False, container=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True, avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录', file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="Embedding model",
                                         value=INIT_EMBEDDING_MODEL)

        # 设置初始化向量数据库按钮的点击事件。当点击时，调用 create_db_info 函数，并传入用户的文件和希望使用的 Embedding 模型。
        init_db.click(create_db_info,
                      inputs=[file, embeddings], outputs=[msg])

        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_with_his_btn.click(model_center.chat_qa_chain_self_answer, inputs=[
                              msg, chatbot,  llm, embeddings, temperature, top_k, history_len],
                              outputs=[msg, chatbot])
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot, llm, embeddings, temperature, top_k], outputs=[msg, chatbot])
        # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        llm_btn.click(respond, inputs=[
                      msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot], show_progress="minimal")

        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(respond, inputs=[
                   msg, chatbot,  llm, history_len, temperature], outputs=[msg, chatbot], show_progress="hidden")
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
    2. 初始化数据库时间可能较长，请耐心等待。
    3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
