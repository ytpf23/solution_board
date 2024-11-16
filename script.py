from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
import os

import os
import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
)

file_name = "pdf.pdf"
extra_info = {"file_name": file_name}

def generate_summary(text, llm):
    # Split long text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Generate a concise summary using CoT
    summary = []
    for chunk in chunks:
        prompt = (
            "Summarize the following text in clear, accessible language for a business audience: \n"
            + chunk
        )
        response = llm.invoke(prompt)
        summary.append(response)
    return "\n\n".join(summary)

with open(f"./{file_name}", "rb") as f:
    # must provide extra_info with file_name key with passing file object
    documents = parser.load_data(f, extra_info=extra_info)
    print(documents)
    print(type(documents))
    print(documents[0].text)
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True)
    print(generate_summary(str(documents[0].text), llm))


