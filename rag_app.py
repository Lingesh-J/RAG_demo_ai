import streamlit as st 
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings # to get embedding models
from langchain_core.documents import Document # to store text and meta data
from langchain_text_splitters import CharacterTextSplitter # to split the large paragraph into small chunks
from langchain_community.vectorstores import FAISS # to store the embedding data fro  the given document for similarity search

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model= genai.GenerativeModel('gemini-2.5-flash-lite')
st.set_page_config('RAG DEMO',page_icon='💻',layout='wide')

def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

with st.spinner('Loading embedding model ⏳'):
    embedding_model=load_embedding()


st.title('RAG Assistant :blue[Using Embedding and LLM]֎🇦🇮')
st.subheader('Your Intelligent Document Assistant 🔍')
    
uploaded_file=st.file_uploader('Upload your file here in PDF format 📤')

if uploaded_file:
    pdf=PdfReader(uploaded_file)
    
    raw_text=''
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    if raw_text.strip():
        # remove spaces and check whether we have tect data and ensure that given raw_text is not empty
        
        doc=Document(page_content=raw_text)
        # to get content in the given pdf and meta data
        
        splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        # max char in each chunk is 1000 and overlap to maintain the relation between context =200
        
        chunk_text= splitter.split_documents([doc])
        # splits the data in document into multiple smaller chunks
        
        text= [ i.page_content for i in chunk_text]
        # to get data as list of smaller text
        
        vector_db=FAISS.from_texts(text,embedding_model)
        
        retrive=vector_db.as_retriever()
        # creates a search tool to find the relevant chunks
        
        st.success('Document processed and saved sucessfully!✅ Ask a question now')
        
        query=st.text_input('Ask me a question')
        
        if query:
            with st.chat_message('human'):
                
                with st.spinner('Analysing the document...'):
                    
                    relevant_data=retrive.invoke(query)
                    # invoke the embedding model and search the similar chunk for the given query
                    
                    content = '\n\n'.join([i.page_content for i in relevant_data])
                    
                    prompt = f'''
                    You are an AI expert. Use the generated content {content} to answer the query {query}. If you are sure
                    with the answer, say "I have no content related to this question. Please ask relevant query to answer"
                    
                    Result in bullet points'''
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown('## :green[Results ⚙️]')
                    st.write(response.text)
                    
    else:
        st.warning('Drop the file in PDF format')