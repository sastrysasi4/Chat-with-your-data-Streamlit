import os
from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone

pinecone.init(      
	api_key=str(os.getenv('Pinecone_api_key')),      
	environment='gcp-starter'      
) 

def ingest_docs()->None:
    loader = ReadTheDocsLoader(path="langchain-docs\latest", encoding="utf-8")
    raw_documents = loader.load()
    print(f'loaded {len(raw_documents)} documents')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 50, separators=["\n\n", "\n"," ",""])
    documents = text_splitter.split_documents(raw_documents)
    print(f"splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace('langchain-docs\latest', "https:/")
        doc.metadata.update({"source": new_url})

    # for windows
    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     replace_path = os.path.join('langchain-docs\latest', "langchain-docs")
    #     new_url = old_path.replace(replace_path, "https:/")
    #     # if url contains '\' replace with '/' (fixes windows path issue)
    #     new_url = new_url.replace("\\", "/")
    #     doc.metadata.update({"source": new_url})

    print(f'going to insert {len(documents)} to Pinecone')

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents,embedding=embeddings,index_name='langchain-pdf-streamlit')
    print("*************** added to pinecone DB ******************")



if __name__ == '__main__':
    ingest_docs()