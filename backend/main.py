import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from .constant import (
    AZURE_SERVICE_ENDPOINT,
    CHATBOT_AZURE_OPENAI_API_VERSION,
    CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME,
    CHATBOT_AZURE_OPENAI_MODEL_NAME,
    EMBEDDING_AZURE_OPENAI_API_VERSION,
    EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME,
    CHATBOT_AZURE_OPENAI_ENDPOINT,
    PINECONE_INDEX_NAME,
)

load_dotenv()


def run_llm(query: str):
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.environ["EMBEDDING_AZURE_OPENAI_API_KEY"],
        azure_deployment=EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_SERVICE_ENDPOINT,
        api_version=EMBEDDING_AZURE_OPENAI_API_VERSION,
    )
    llm = AzureChatOpenAI(
        api_version=CHATBOT_AZURE_OPENAI_API_VERSION,
        azure_deployment=CHATBOT_AZURE_OPENAI_DEPLOYMENT_NAME,
        model_name=CHATBOT_AZURE_OPENAI_MODEL_NAME,
        api_key=os.environ["CHATBOT_AZURE_OPENAI_API_KEY"],
        azure_endpoint=CHATBOT_AZURE_OPENAI_ENDPOINT,
    )
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )

    qa = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain?")
    print(res["answer"])
