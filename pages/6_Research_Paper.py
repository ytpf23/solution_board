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

#load the LLAMA API key
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sidebar for API key input
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="api_key", type="password")
#     llama_api_key = st.text_input("LLamaParse API Key", key="llama_key", type="password")

# Set up Streamlit page
st.title("TechCombinator")
st.caption("Connecting industry problems to scientific solutions powered by AI")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
def generate_summary(text, llm):
    # Split long text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Generate a concise summary using CoT
    summary = []
    for chunk in chunks:
        prompt = (
            "Summarize the following text in 5 sentences in clear, accessible language for a business audience: \n"
            + chunk
        )
        response = llm.invoke(prompt)
        summary.append(response.content)
    return "\n\n".join(summary)

# Function to ask targeted questions for refinement
def ask_followup_question(summary, llm):
    prompt = (
        "Based on the initial summary below, propose a question that would help "
        "clarify any ambiguities or gather additional context?\n"
        + summary
    )
    response = llm.invoke(prompt)
    return response.content

# Handle file upload
uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
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
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, streaming=True)

    # Extract relevant application summary
    with st.spinner("Generating summary..."):
        initial_summary = generate_summary(research_text, llm)
    st.subheader("AI-Generated Summary")
    st.write(initial_summary)

    # Generate buiness 
    with st.spinner("Generating summary..."):
        initial_summary = generate_summary(research_text, llm)
    st.subheader("AI-Generated Summary")
    st.write(initial_summary)

    # Store the summary in session state for conversation
    st.session_state["messages"].append({"role": "assistant", "content": initial_summary})

    # Ask follow-up question for clarification
    with st.spinner("Asking follow-up question..."):
        question = ask_followup_question(initial_summary, llm)
    st.subheader("Follow-up Question")
    st.write(question)

    # Capture researcher's response
    researcher_response = st.text_input("Your answer to the question:")
    if researcher_response:
        st.session_state["messages"].append({"role": "user", "content": researcher_response})
        st.write("Response recorded.")

        # Refine the summary with additional input
        prompt = (
            "Refine the summary using the original summary, the follow-up question, "
            "and the researcher's response:\n\n"
            f"Original Summary: {initial_summary}\n\n"
            f"Researcher's Response: {researcher_response}"
        )
        refined_summary = llm.invoke(prompt).content
        st.subheader("Refined Summary")
        st.write(refined_summary)

        # Save the refined summary to conversation history
        st.session_state["messages"].append({"role": "assistant", "content": refined_summary})

# Display conversation history
st.subheader("Conversation History")
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
