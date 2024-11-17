import os
import random
import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from llama_parse import LlamaParse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

# load the LLAMA API key
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class params:
    psql_host = os.getenv(
        "PSQL_HOST",
        "team-decision-instance-1.ch2w6gwwm7no.eu-central-1.rds.amazonaws.com",
    )
    psql_port = int(os.getenv("PSQL_PORT", "5432"))
    psql_user = os.getenv("PSQL_USER", "techcombinator")
    psql_password = os.getenv("PSQL_PASSWORD")
    psql_db = os.getenv("PSQL_DB", "postgres")
    psql_schema = os.getenv("PSQL_SCHEMA", "techcombinator")
    psql_conn_string = "postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}?options=-csearch_path%3Ddbo,{5}".format(
        psql_user, psql_password, psql_host, psql_port, psql_db, psql_schema
    )
    collection_name = os.getenv("COLLECTION NAME", "test_paper")


# Sidebar for API key input
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="api_key", type="password")
#     llama_api_key = st.text_input("LLamaParse API Key", key="llama_key", type="password")

# Set up Streamlit page
st.title("SciCombinator")
st.caption("Connecting industry problems to scientific solutions powered by AI")

# Initialize conversation history
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []


# Function to extract text using LLamaParse
def extract_text_with_llamaparse(pdf_file):
    bytes_data = pdf_file.getvalue()
    # st.write(bytes_data)

    parser = LlamaParse(
        api_key=LLAMA_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )
    documents = parser.load_data(bytes_data, extra_info={"file_name": "_"})
    all_text = documents[0].text
    return all_text


# Function to generate a concise summary using CoT approach
def find_relevant_sections(text, llm):
    # Split long text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Find relevant sections in a paper that can relate to potential business usecases
    relevant_sections = []
    for chunk in chunks:
        prompt = (
            "Let's think step by step. Find the most relevant sections in the following research article that can relate to potential business usecases and summarize them in clear, accessible language for a business audience: \n"
            + chunk
        )
        response = llm.invoke(prompt)
        relevant_sections.append(response.content)
    return "\n\n".join(relevant_sections)


# Function to generate a concise summary using CoT approach
def generate_summary(text, llm):
    # Split long text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Generate a concise summary using CoT
    summary = []
    for chunk in chunks:
        prompt = (
            "Be concise. Summarize the following text in 5 sentences in clear, accessible language for a business audience. Translate the research jargon into understandable concepts. :\n"
            + chunk
        )
        response = llm.invoke(prompt)
        summary.append(response.content)
    return "\n\n".join(summary)


# Function to generate a concise summary using CoT approach
def extract_metadata(text, llm):
    # Split long text into manageable chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    # chunks = text_splitter.split_text(text)

    # Generate a concise summary using CoT
    summary = []
    prompt = (
        "Extract the authors, title and one authors email from the research paper. \n"
        + "Provide the answer in a form (AUTHORS: ...; TITLE: ...; EMAIL: ...)"
        + "For example the proper answer would be (AUTHORS: Lukasz Sztukiewicz, Artur Dubrawski; TITLE: Exploring Loss Design Techniques; EMAIL: research@lukaszsztukiewicz.com)"
        + "Extract the authors, title and one authors email from this RESARCH PAPER: \n"
        + text
        + "\n"
        + "(AUTHORS: "
    )
    response = llm.invoke(prompt)
    summary.append(response.content)
    return "\n\n".join(summary)


def clarify_summary(summary, CLARIFICATION, llm):
    prompt3 = (
        "Based on the CLARIFICATION, rewrite the SUMMARY so it includes all relevant information. The text should be easy to read and concise. \n"
        + f"CLARIFICATION: {CLARIFICATION}"
        + "\n"
        + "SUMMARY: "
        + summary
    )
    response3 = llm.invoke(prompt3)
    return response3.content


def upload_doc(refined_summary, title, authors, email):
    print("Uploading document")
    document = create_pg_document_from_pdf_refined_summary(
        refined_summary, title, authors, email
    )
    print("Document created")
    print(document)


# Function to ask targeted questions for refinement
def ask_followup_question(summary, research_text, llm):
    prompt = (
        "Based on the business summary of a research paper below, propose a question that would help "
        "clarify any ambiguities or gather additional context?\n" + summary
    )
    response = llm.invoke(prompt)
    prompt2 = (
        "Based on the RESAERCH PAPER, answer the following QUESTION: \n"
        + f"QUESTION: {response.content}"
        + "\n"
        + "RESARCH PAPER: "
        + research_text
        + "\n"
        + "ANSWER: "
    )
    clarification = llm.invoke(prompt2)

    return clarify_summary(summary, clarification.content, llm)


def handle_refine_summary(refined_summary, researcher_clarification, llm, key):
    with st.spinner("Clarifying the summary..."):
        refined_summary = clarify_summary(
            refined_summary, researcher_clarification, llm
        )
    st.subheader("Refined summary", key=key)
    st.write(refined_summary, key=key)
    st.session_state["clarifications"] += 1
    return refined_summary


def create_pg_document_from_pdf_refined_summary(refined_summary, title, authors, email):
    mock = "Intuitive Surgical faces challenges in ensuring precise, real-time 3D reconstruction of anatomical structures during robotic-assisted surgeries, which is crucial for enhancing surgical accuracy. Current methods can struggle with surface discontinuities and inconsistencies, leading to potential inaccuracies in guiding robotic instruments. These limitations may impact surgical precision, increase the risk of tissue damage, and hinder optimal patient outcomes."
    document = Document(
        page_content=refined_summary + mock,
        metadata={
            "id": str(42),
            "title": title,
            "authors": authors,
            "year": "mockyear",
            "venue": "mockvenue",
            "n_citation": "mock_n_citation",
            "references": "mockyear_references",
            "email": email,
        },
    )
    documents = [document]
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    return PGVector.from_documents(
        documents=documents,
        collection_name=params.collection_name,
        connection=params.psql_conn_string,
        embedding=hf_embeddings,
        pre_delete_collection=False,
        use_jsonb=True,
    )


# Handle file upload
uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
st.session_state["clarifications"] = 0
if uploaded_file:
    if not OPENAI_API_KEY:
        # st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        st.warning("Did not found OPENAI KEY in env.")
        st.stop()

    # Extract text from uploaded PDF using LLamaParse
    with st.spinner("Extracting text from PDF..."):
        research_text = extract_text_with_llamaparse(uploaded_file)
    st.success("Research paper uploaded successfully.")

    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, streaming=True
    )

    # Generate buiness summary
    with st.spinner(
        "Extracting the authors, title and one authors email from the research paper ..."
    ):
        extracted_metadata = extract_metadata(research_text, llm)
    st.subheader("Extracted metadata")

    try:
        authors = extracted_metadata.split(";")[0]
    except:
        authors = "Not found"

    try:
        title = extracted_metadata.split(";")[1].split(":")[1].strip()
    except:
        title = "Not found"

    try:
        email = extracted_metadata.split(";")[2].split(":")[1].strip()
    except:
        email = "Not found"

    st.write("AUTHORS: ", authors)
    st.write("TITLE: ", title)
    st.write("EMAIL: ", email)

    # Extract relevant application summary
    with st.spinner("Finding relevant sections..."):
        relevant_sections_summary = find_relevant_sections(research_text, llm)
    st.subheader("Summary of relevant sections for business usecases")
    st.write(relevant_sections_summary)

    # Generate buiness summary
    with st.spinner("Translating the research jargon ..."):
        initial_summary = generate_summary(relevant_sections_summary, llm)
    st.subheader("Business ready summary")
    st.write(initial_summary)

    # Store the summary in session state for conversation
    # st.session_state["messages"].append({"role": "assistant", "content": extracted_metadata})
    # st.session_state["messages"].append({"role": "assistant", "content": relevant_sections_summary})
    # st.session_state["messages"].append({"role": "assistant", "content": initial_summary})

    # Ask follow-up question for clarification
    with st.spinner("Clarifying the summary..."):
        clarified_summaary = ask_followup_question(initial_summary, research_text, llm)
    st.subheader("Clarified summary")
    st.write(clarified_summaary)

    # Capture researcher's response
    refined_summary = clarified_summaary
    key = str(st.session_state["clarifications"])
    researcher_clarification = st.text_input(
        "Write things you want to add or reformulate in the summary:", key=key
    )

    st.button(
        "Accept summary",
        on_click=lambda: upload_doc(refined_summary, title, authors, email),
    )

    st.button(
        "Reformulate summary",
        on_click=lambda: handle_refine_summary(
            refined_summary, researcher_clarification, llm, key
        ),
    )


# Display conversation history
# st.subheader("Conversation History")
# for msg in st.session_state["messages"]:
#     st.chat_message(msg["role"]).write(msg["content"])
