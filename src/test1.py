import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import hashlib

# Load environment variables
load_dotenv()

# Initialize a simple database for users
if "users" not in st.session_state:
    st.session_state.users = {"Admin": hashlib.sha256("1234".encode()).hexdigest()}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

# Hash passwords for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication functions
def sign_up():
    st.subheader("Sign Up")
    username = st.text_input("Enter a username")
    password = st.text_input("Enter a password", type="password")
    confirm_password = st.text_input("Confirm your password", type="password")
    if st.button("Sign Up"):
        if not username or not password:
            st.error("Username and password cannot be empty.")
        elif username in st.session_state.users:
            st.error("Username already exists. Please choose a different username.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            st.session_state.users[username] = hash_password(password)
            st.success("Account created successfully! Please log in.")

def login():
    st.subheader("Log In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        hashed_password = hash_password(password)
        if username in st.session_state.users and st.session_state.users[username] == hashed_password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Invalid username or password.")

# Authentication page handler
def auth_page():
    st.title("Welcome to SAARANSH")
    auth_choice = st.radio("Choose an option", ["Log In", "Sign Up"])
    if auth_choice == "Log In":
        login()
    elif auth_choice == "Sign Up":
        sign_up()


# CSV functionalities (kept unchanged from original code)
def get_csv_data(csv_files):
    try:
        data_frames = []
        for csv in csv_files:
            df = pd.read_csv(csv)
            data_frames.append(df)
        return data_frames
    except Exception as e:
        st.error(f"Error reading CSV files: {e}")
        return []

def plot_graph(df, x_col, y_col, graph_type):
    try:
        plt.figure(figsize=(10, 6))
        if graph_type == "Line Plot":
            plt.plot(df[x_col], df[y_col], marker='o')
        elif graph_type == "Bar Plot":
            plt.bar(df[x_col], df[y_col], color='skyblue')
        elif graph_type == "Scatter Plot":
            plt.scatter(df[x_col], df[y_col], color='green', edgecolor='black')
        elif graph_type == "Histogram":
            plt.hist(df[y_col], bins=20, color='orange', edgecolor='black')
        elif graph_type == "Pie Chart":
            if df[y_col].dtype in [float, int] and len(df[x_col]) == len(df[y_col]):
                plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%', startangle=140)
            else:
                st.error("Pie Chart requires numeric values for 'y_col' and matching dimensions for 'x_col' and 'y_col'.")
                return
        if graph_type != "Pie Chart":
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{graph_type} of {y_col} vs {x_col}")
            plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error generating graph: {e}")

# PDF functionalities
def get_pdf_text(pdf_files):
    text = ""
    try:
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error while extracting text from PDF: {e}")
    return text

def get_text_chunks(text):
    try:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error while splitting text into chunks: {e}")
        return []

def get_vectorstore_from_pdf(pdf_files):
    try:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error while creating vector store from PDF: {e}")

def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error while creating vector store from URL: {e}")

# Conversation chain
def create_conversation_chain(vectorstore): 
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        if hasattr(vectorstore, "as_retriever"):
            retriever = vectorstore.as_retriever()
        else:
            st.error("Vectorstore does not support retrieval.")
            return None

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        conversation_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
        return conversation_chain
    except Exception as e:
        st.error(f"Error while creating conversation chain: {e}")

def is_query_relevant(query, vectorstore):
    return True  # Placeholder logic, adjust as needed

# Main application
def main():
    st.set_page_config(page_title="Chat with documents and CSVs", page_icon="📊")

    if not st.session_state.authenticated:
        auth_page()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.experimental_rerun()

        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap'); 

        .title {
          font-family: 'Bebas Neue', cursive; 
          font-size: 45px;
          color: #958F9A ;
          text-align: center;
          transition: text-shadow 0.3s ease-in-out;   
        }

        .title:hover {
          text-shadow: 0 0 45px #6C02B9 ;
        }
        </style>

        <h1 class='title'>SAARANSH</h1>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.header("Settings")
            chat_mode = st.radio("Choose chat mode:", ["Chat with Website", "Chat with PDFs", "Chat with CSV"])
            if chat_mode == "Chat with Website":
                website_url = st.text_input("Website URL")
                pdf_files = None
                csv_files = None
            elif chat_mode == "Chat with PDFs":
                website_url = None
                csv_files = None
                pdf_files = st.file_uploader("Upload your PDFs here", type=['pdf'], accept_multiple_files=True)
            else:
                website_url = None
                pdf_files = None
                csv_files = st.file_uploader("Upload your CSV files here", type=['csv'], accept_multiple_files=True)

        if chat_mode == "Chat with Website":
            if not website_url:
                st.error("Please enter a website URL")
                return
            vectorstore = get_vectorstore_from_url(website_url)
        elif chat_mode == "Chat with PDFs":
            if not pdf_files:
                st.error("Please upload PDF files")
                return
            vectorstore = get_vectorstore_from_pdf(pdf_files)
        elif chat_mode == "Chat with CSV":
            if not csv_files:
                st.error("Please upload CSV files")
                return
            data_frames = get_csv_data(csv_files)
            if data_frames:
                st.subheader("Uploaded Data")
                for i, df in enumerate(data_frames):
                    st.write(f"DataFrame {i+1}")
                    st.dataframe(df)

                    st.subheader("Visualize Data")
                    columns = df.columns.tolist()
                    x_col = st.selectbox(f"Select X-axis for DataFrame {i+1}", columns)
                    y_col = st.selectbox(f"Select Y-axis for DataFrame {i+1}", columns)
                    graph_type = st.radio(f"Graph Type for DataFrame {i+1}", ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Pie Chart"])

                    if st.button(f"Plot Graph for DataFrame {i+1}"):
                        plot_graph(df, x_col, y_col, graph_type)
            return

        if vectorstore is None:
            st.error("Failed to create vectorstore.")
            return

        conversation_chain = create_conversation_chain(vectorstore)
        if conversation_chain is None:
            return

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

        user_query = st.chat_input("Type your message here...")
        if user_query:
            try:
                if is_query_relevant(user_query, vectorstore):
                    response = conversation_chain.invoke({"chat_history": st.session_state.chat_history, "input": user_query})
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=response['answer']))
                else:
                    st.error("Sorry, I can only respond to questions related to the provided content.")
            except Exception as e:
                st.error(f"Error while processing user query: {e}")

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                st.markdown(
                    f"<div style='background-color: #3b3a39; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>Rudra🤖<br /><br />{message.content}</div>",
                    unsafe_allow_html=True)
            elif isinstance(message, HumanMessage):
                st.markdown(
                    f"<div style='background-color: #935DC6 ; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>You🙋<br /><br />{message.content}</div>",
                    unsafe_allow_html=True)

if __name__ == "__main__":
    main()