# LangChain Documentation Assistant
Want to interact with your documents like you're chatting with an expert? LangChain Documentation Assistant is the solution for you. This full-stack web application uses AI to help you easily search, extract information, and work with your documents more effectively.

## Demo
This project uses documents crawled from "https://python.langchain.com/docs/introduction/", then extracts and ingests them into Pinecone.

## Technology
Backend
- Language: Python
- Framework: Langchain
- Chat api: Azure openai
- Crawl data: Tavily
- Vector store: Pinecone

Frontend
- Streamlit

# How to build project
Init project and Install packages:
- uv init --python 3.12
- uv add langchain langchain-community langchain-openai langchain-pinecone langchain-tavily langchainhub langsmith python-dotenv streamlit

Enviroments:
- Login azure and create 2 api: chat api "gpt-4o-mini", embed api "text-embedding-3-small"
- Login langsmith, tavily, pipecone
-> copy and pasted keys, models,... into .env file and constant file

Run backend
- python backend/main.py

Run Frontend 
- streamlit run frontend.py