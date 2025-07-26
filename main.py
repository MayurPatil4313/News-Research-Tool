import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import tqdm as notebook_tqdm
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import pickle
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
import langchain
import time

# streamlit run main.py --server.port 8502
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("API_KEY")
# print('======================================================',API_KEY)

st.title("News Research Tool  📈")

st.sidebar.title("News Article URLs ")

file_path = "faiss_store_vector_index.pkl"

main_placeholder = st.empty()
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.9,
    max_tokens=500,
    groq_api_key=API_KEY
)

url_inputs = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder="Enter URL")
    url_inputs.append(url)

process_urls_clicked = st.sidebar.button("Process URLs")

if process_urls_clicked:

    if len(url_inputs[0]) or len(url_inputs[1]) != 0:
        # load the data
        loader = UnstructuredURLLoader(urls=url_inputs)
        main_placeholder.text("Data Loading started......")
        data = loader.load()

        # split the data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", "."], chunk_size=1000, chunk_overlap=200
        )

        main_placeholder.text("Data Splitting.....")
        docs = text_splitter.split_documents(data)

        # create embedding
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        main_placeholder.text("Data Embedding......")

        vectorindex = FAISS.from_documents(documents=docs, embedding=embeddings)

        main_placeholder.text("Data Embedding Done ✅✅✅")

        # storing FASSI index into pickle format
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex, f)

    else:
        st.error("Provide the URLs First")


query = st.text_input("Question:🔍", placeholder="Enter Your query")

if st.button("Search 🔍"):
    print(query)
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorindex = pickle.load(f)

                # create a QA chain
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, retriever=vectorindex.as_retriever()
                )

                result = chain({'question':query} ,return_only_outputs=True)
                st.header("╰┈➤")
                st.write(result['answer'].split(':')[1])

                # display source 
                sources = result.get('sources' , '')

                if sources:
                    st.subheader("🌐 Sources")
                    source_list = sources.split('\n')

                    for source in source_list:
                        st.write(source)

    else:
        st.error("Provide the Question First")