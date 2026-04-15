# RAG_demo_ai
RAG algorithm using Hugging face embedding
### RAG algorithm using hugging face embedding- Trained only on given data

RAG is a technique that enhances the capablities of LLMs by combining the information retrival with the text generation, 
Instead of replying on pre-trained knowledge, RAG fetch the relevant data from external sources and use it to generate more accurate responses.

### Packages

streamlit
python-dotenv
google-generativeai
PyPDF2



langchain # core frameworks
langchain-community # extra integration
faiss-cpu # fast vector database to store the embedded data
langchain-huggingface # connect huggingface models to perform embedding
langchain-text-splitters # to split the data into chunks
sentence-transformers # pre trained models to convert chunks into vectors
langchain-core # to handle documents, chains of datahain # core frameworks
langchain-community # extra integration
faiss-cpu # fast vector database to store the embedded data
langchain-huggingface # connect huggingface models to perform embedding
langchain-text-splitters # to split the data into chunks
sentence-transformers # pre trained models to convert chunks into vectors
langchain-core # to handle documents, chains of data