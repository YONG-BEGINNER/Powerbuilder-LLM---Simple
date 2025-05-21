import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from llm.llm_template import prompt_template ,refine_template

# Load environment variables
load_dotenv()

# Initialize Groq client and LangChain LLM
ChatGroq.api_key = os.environ['GROQ_API_KEY']
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model_use = "llama3-70b-8192"

# Define a function to get completion from Groq model
def get_completion(prompt, model=model_use):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Load embedding model and vector store
persist_directory = './database/vector_db/chroma'
embedding = HuggingFaceEmbeddings(model="moka-ai/m3e-base")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(f"Vectors Quantity: {vectordb._collection.count()}")

refine_prompt_template = refine_template("v2")
doc_qa_template = prompt_template("v1")

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=doc_qa_template)


# Initialize memory and QA chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm=ChatGroq(model_name=model_use, temperature=0.3),
                                           retriever=vectordb.as_retriever(),
                                           memory=memory)

question = ["""
What is the software that this the context talking about?
A. Python B.Power BI C.PowerBuilder D.PowerBuilder
"""]

for q in question:
#     refined_question = get_completion(refine_prompt_template.format(question=q))
#     print(f"\nRefined Question: \n{refined_question}")
    result = qa.invoke({'question': q})
    print(f"LLM + Document Answer:\n{result['answer']}\n")
