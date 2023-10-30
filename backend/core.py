import os 
from dotenv import load_dotenv
load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from typing import Any, Dict, List

pinecone.init(      
	api_key=str(os.getenv('Pinecone_api_key')),      
	environment='gcp-starter'      
) 
# index = pinecone.Index('langchain-pdf-streamlit')

def run_llm(query:str):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name='langchain-pdf-streamlit',embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm = chat, chain_type="stuff",retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa({"query": query})



if __name__=='__main__':
    print(run_llm(query='what is RetrievalQA chain?'))