import asyncio
import os
import re
from dotenv import load_dotenv
from langchain_tavily import TavilyCrawl
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from constant import (
    AZURE_SERVICE_ENDPOINT,
    EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME,
    EMBEDDING_AZURE_OPENAI_API_VERSION,
    PINECONE_INDEX_NAME,
    target_url,
)

load_dotenv()


async def crawl(target_url):
    """
    Crawls documentation pages from a target URL using TavilyCrawl.
    """
    # print(f"Start crawling data from URL: {target_url}")
    crawler = TavilyCrawl(
        max_depth=2,
        limit=10,
        instructions="Crawl all documentation pages on langchain",
    )
    crawl_results = await crawler.ainvoke({"url": target_url})
    # print(f"Crawled {len(crawl_results)} pages.")
    return crawl_results


def extract_data(crawl_results):
    """
    Extracts structures the crawled data into Document objects.
    """
    # print(type(crawl_results)) # <class 'dict'>
    # print(crawl_results.keys()) # dict_keys(['base_url', 'results', 'response_time', 'request_id'])
    all_docs = []
    for crawl_result_item in crawl_results["results"]:
        # print(crawl_result_item.keys()) # dict_keys(['url', 'raw_content'])
        all_docs.append(
            Document(
                page_content=crawl_result_item["raw_content"],
                metadata={"source": crawl_result_item["url"]},
            )
        )
    return all_docs


async def ingest_docs(docs):
    """
    Splits documents and ingests them into a Pinecone vector store.
    """
    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    print(f"Add {len(documents)} chunks to Pinecone")

    # store
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.environ["EMBEDDING_AZURE_OPENAI_API_KEY"],
        azure_deployment=EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_SERVICE_ENDPOINT,
        api_version=EMBEDDING_AZURE_OPENAI_API_VERSION,
    )
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=PINECONE_INDEX_NAME
    )
    print("****Loading to vectorstore done ***")


async def main():
    """
    Main function to run the entire data pipeline.
    """
    crawled_data = await crawl(target_url)
    cleaned_data = extract_data(crawled_data)
    await ingest_docs(cleaned_data)


if __name__ == "__main__":
    asyncio.run(main())
