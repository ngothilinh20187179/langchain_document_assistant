from typing import Set
from backend.main import run_llm
import streamlit as st

st.header("LangChain Document Assistant")
# Create input (UI)
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

# Initialize the Session State
# Check if "user_prompt_history", "chat_answers_history" exist in state ? If not - initializes them
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    """
    Format document links of answer
    """
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)
        question, answer, source_documents = (
            generated_response["input"],
            generated_response["answer"],
            generated_response["context"],
        )

        # for doc in source_documents:
        #     print(doc.metadata["source"])

        sources = set([doc.metadata["source"] for doc in source_documents])
        formatted_response = f"{answer} \n\n {create_sources_string(sources)}"

        # add the formatted question and answer into state every time submit a prompt and receive an answer from the LLM
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

# check if any chat history then loops through the two lists
# display each pair of questions-and-answers
if st.session_state["chat_answers_history"]:
    for chat_answer, user_question in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(chat_answer)


# Output
# {
#   "input": "What is LangChain?",
#   "answer": "LangChain is a framework that implements a standard interface for large language models and related technologies, such as embedding models and vector stores. It integrates with hundreds of providers, allowing developers to easily combine various components for building applications that utilize advanced language processing capabilities. LangChain provides an orchestration framework called `langgraph`, which facilitates the creation of production-ready applications with features like persistence and streaming.",
#   "context": [
#     {
#       "id": "6fbe821f-8bf6-4b63-825c-0ba7f095c202",
#       "metadata": {
#         "source": "https://python.langchain.com/docs/concepts/chat_models"
#       },
#       "page_content": "More\n\n* [Homepage](https://langchain.com/)\n* [Blog](https://blog.langchain.dev/)\n* [YouTube](https://www.youtube.com/@LangChain)\n\nCopyright © 2025 LangChain, Inc."
#     },
#   ]
# }
