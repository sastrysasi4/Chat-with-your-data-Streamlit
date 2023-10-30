
# Chat with your Data - Streamlit  

In this project, we will learn how to use Streamlit, Langchain, and chat with your HTML documents with memory.

## Installation

These are the required packages

* Langchain
* OpenAI
* Streamlit
* Streamlit-Chat
* Beautifulsoup4
* Pinecone-Client
* Tiktoken
## Roadmap

- **Ingestion.py** : creates the HTML data into vectors in the Pinecone vector Database

- **Core.py** : ConversationalRetrievalChain for chat history

- **Main.py** : Streamlit UI integration



## Documentation

We'll see an end-to-end document chat bot with UI here. In this project, we are using HTML files that contain information on Langchain that was scraped from the website as data. We use the Pinecone client server for vector DB, and Streamlit and Streamlit-Chat for UI and chat UI. We've also given the chatbot memory, so it can recall earlier conversations. which is ConversationalRetrievalChain.


## Features

* Model -- OpenAI API
* Embeddings -- OpenAI API
* Document loader -- ReadTheDocsLoader (HTML)
* Text splitter -- RecursiveCharacterTextSplitter
* Vector DB -- Pinecone
* QA chain -- ConversationalRetrievalChain


## Authors

- [@sasidhar](https://github.com/sastrysasi4)