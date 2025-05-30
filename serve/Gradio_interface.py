import gradio as gr
import os

db_with_his_btn = gr.Button("Chat db with history")
db_wo_his_btn = gr.Button("Chat db without history")

def generate(input, temperature):
    
    output = llm.invoke(input, temperature)
    return output

def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"

    prompt = f"{prompt}\nUser:{message}\nAssistant:"

    return prompt

def response(message, chat_history):
    formatted_prompt = format_chat_prompt(message, chat_history)
    bot_message = llm.invoke(formatted_prompt,
                             max_new_token=1024,
                             stop_sequences=["\nUser:", ""])
    
    chat_history.append((message, bot_message))
    return "",chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value=[msg,chatbot])

    btn.click(response, inputs=[msg,chatbot], outputs=[msg,chatbot])
    msg.submit(response, inputs=[msg,chatbot], outputs=[msg,chatbot])

gr.close_all()

demo.launch()