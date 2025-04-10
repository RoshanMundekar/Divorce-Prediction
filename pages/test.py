import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.text_splitter import CharacterTextSplitter
import openai
from dotenv import load_dotenv
load_dotenv()
# Set OpenAI API Key
# Get OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key is missing. Please set it in the `.env` file.")
# Document file paths
file1 = "./data/DIVISION OF ASSETS AFTER DIVORCE.txt"
file2 = "./data/INHERITANCE.txt"

# Function to initialize the OpenAI embeddings and model
def openai_setting():
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    model_name = "gpt-4o"
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
    return embedding, llm

# Function to split the law content
def law_content_splitter(path, splitter="CIVIL CODE"):
    with open(path, encoding="utf-8") as f:
        law_content = f.read()
    law_content_by_article = law_content.split(splitter)[1:]
    text_splitter = CharacterTextSplitter()
    return text_splitter.create_documents(law_content_by_article)

# Function to handle chatbot logic
def chatbot1(question):
    try:
        return agent.run(question)
    except Exception as e:
        return f"I'm sorry, I'm having trouble understanding your question. Error: {str(e)}"

# List of greetings
greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "whats up"]

# Function to determine if input is a greeting
def is_greeting(input_str):
    return any(greet in input_str.lower() for greet in greetings)

# Function to handle chatbot logic
def chatbot(input_str):
    if any(input_str.lower().startswith(greet) for greet in greetings):
        if len(input_str.split()) <= 3:
            return "Hello! Ask me your question about Italian Divorce or Inheritance Law?"
        else:
            return chatbot1(input_str)
    else:
        return chatbot1(input_str)

# Splitting the content of law documents
divorce_splitted = law_content_splitter(file1)
inheritance_splitted = law_content_splitter(file2)

# Initializing embedding and language model
embedding, llm = openai_setting()

# Define the prompts
divorce_prompt = """As a specialized bot in divorce law, you should offer accurate insights on Italian divorce regulations.
You should always cite the article numbers you reference.
{context}

Question: {question}"""
DIVORCE_BOT_PROMPT = PromptTemplate(template=divorce_prompt, input_variables=["context", "question"])

inheritance_prompt = """As a specialist in Italian inheritance law, you should deliver detailed and accurate insights.
Always cite the article numbers you reference.
{context}

Question: {question}"""
INHERITANCE_BOT_PROMPT = PromptTemplate(template=inheritance_prompt, input_variables=["context", "question"])

# Setup for Chroma databases and RetrievalQA
chroma_directory = "./docs/chroma"

inheritance_db = Chroma.from_documents(
    documents=inheritance_splitted,
    embedding=embedding,
    persist_directory=chroma_directory,
)
inheritance = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=inheritance_db.as_retriever(),
    chain_type_kwargs={"prompt": INHERITANCE_BOT_PROMPT},
)

divorce_db = Chroma.from_documents(
    documents=divorce_splitted, embedding=embedding, persist_directory=chroma_directory
)
divorce = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=divorce_db.as_retriever(),
    chain_type_kwargs={"prompt": DIVORCE_BOT_PROMPT},
)

# Define the tools for the chatbot
tools = [
    Tool(
        name="Divorce Italian law QA System",
        func=divorce.run,
        description="Useful for answering questions about divorce laws in Italy.",
    ),
    Tool(
        name="Inheritance Italian law QA System",
        func=inheritance.run,
        description="Useful for answering questions about inheritance laws in Italy.",
    ),
]

# Initialize conversation memory and agent
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key="output")
react = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent = AgentExecutor.from_agent_and_tools(tools=tools, agent=react.agent, memory=memory, verbose=False)

# Streamlit UI Setup
def setup_ui():
    st.set_page_config(page_title="Analysis of Marital Dynamics and Divorce Prediction", page_icon="⚖️")
    st.title("🏛️ AI-Driven:  Analysis of Marital Dynamics and Divorce Prediction ")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello, I here to help you with Divorce Law"}]

    # Display previous messages and handle new user input
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask your question in English😉"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response = chatbot(user_input)
            response_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    setup_ui()
