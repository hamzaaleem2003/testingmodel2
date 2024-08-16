import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import time
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import  RetrievalQA
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Retrieve API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class ChatBot():
    embeddings = HuggingFaceEmbeddings(fffddffc)
    index_name = "docs-rag-chatbot"

    # Initialize Pinecone with the API key from environment variables
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    
    knowledge = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    template = """
    this is the data from the book "Jami-Khalq-par-Huzoor-ki-Rahmat-o-Shafqat" written by huzoor sheikh ul islam dr muhammad tahir ul qadri, whatever question is asked you have to answer that properly and comprehensively. Whatever question is asked you have to answer properly and comprehensively also, whenever a question is asked from this book you always have to answer the question in Urdu language no matter if in prompt it mentions to answer in Urdu or not, but if it specifies to answer in some other language, only then you have to change the language in giving a response.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": knowledge.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

bot = ChatBot()


st.set_page_config(page_title="Urdu Book Bot")
with st.sidebar:
    st.title('Urdu Book Bot')

# Function for generating LLM response incrementally
def generate_response_stream(input):
    response = bot.rag_chain.invoke(input)
    # Simulate streaming by yielding one character at a time
    for char in response:
        yield char
        time.sleep(0.005)  # Adjust this to control the typing speed

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask me anything from this book"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_container = st.empty()  # Create an empty container for streaming the response
        response_text = ""

        for char in generate_response_stream(input):
            response_text += char
            response_container.write(response_text)

    message = {"role": "assistant", "content": response_text}
    st.session_state.messages.append(message)
