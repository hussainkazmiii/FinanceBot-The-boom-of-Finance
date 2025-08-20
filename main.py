import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables

load_dotenv()

st.header("FinanceBot: The boom of Finance. ")
st.sidebar.header("Input your URLS related to finance: ")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm_placeholder = st.empty()

# Get the API key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please create a .env file with your key.")
    st.stop()

# Initialize LLM outside the processing block for better performance
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0.7
)

if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.error("Please enter at least one valid URL")
        st.stop()
    
    try:
        # load data
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        if not data:
            st.error("No data could be loaded from the provided URLs")
            st.stop()
            
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200  # Added overlap for better context
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        
        # create embeddings and save it to FAISS index
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",  # Better alternative
            model_kwargs={'device': 'cpu'},  # or 'cuda' if available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        vector_index = FAISS.from_documents(docs, embedding_model)
        
        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vector_index, f)
            
        st.success("Processing completed successfully!")
        time.sleep(2)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # Return top 3 results
                )
                result = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
    else:
        st.warning("Please process URLs first before asking questions")