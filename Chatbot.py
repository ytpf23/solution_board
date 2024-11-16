from openai import OpenAI
import streamlit as st
from typing import List, Tuple

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from langchain.agents import initialize_agent, AgentType, create_openai_tools_agent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 

from langchain.globals import set_debug, set_verbose
set_verbose(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("TechCombinator")
st.caption("Conncecting industry problems to scientific solutions powered by the Artificial Intelligence")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, which company do you come from?"}]

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
class params:
    psql_host = os.getenv("PSQL_HOST", "team-decision-instance-1.ch2w6gwwm7no.eu-central-1.rds.amazonaws.com")
    psql_port = int(os.getenv("PSQL_PORT",'5432'))
    psql_user = os.getenv("PSQL_USER", "techcombinator")
    psql_password = os.getenv("PSQL_PASSWORD")
    psql_db = os.getenv("PSQL_DB", "postgres")
    psql_schema = os.getenv("PSQL_SCHEMA", "techcombinator")
    psql_conn_string='postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}?options=-csearch_path%3Ddbo,{5}'.format(psql_user,psql_password,psql_host,psql_port,psql_db,psql_schema)
    collection_name = os.getenv("COLLECTION NAME", "test_paper")

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def create_research_agent(vector_db):
    @tool(return_direct=True)
    def paper_search(query: str) -> List[dict]:
        """
        Search for relevant papers. Returns list of paper details.
        Args:
            query: Search query string
        Returns:
            List of papers with their details
        """
        results = vector_db.similarity_search_with_score(query, k=5)
        return [{
            'title': doc.metadata['title'],
            'abstract': doc.page_content[:500],
            'citations': doc.metadata['n_citation']
        } for doc, score in results]
    @tool
    def generate_business_explanation(paper_details: dict) -> str:
        """Generate business relevance explanation for a paper."""
        return f"Business relevance for {paper_details['title']}"
    @tool
    def get_user_info(query: str) -> str:
        """Process user query and return refined search terms."""
        return query
    # Rest of your agent setup...
    tools = [paper_search, generate_business_explanation, get_user_info]
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Tech Combinator. Use the tools to: 1) understand user needs, 2) find papers, 3) explain relevance"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

vector_db = PGVector(
    collection_name=params.collection_name,
    connection=params.psql_conn_string,
    embeddings=hf_embeddings
)


agent_executor = create_research_agent(vector_db)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent(
        [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        # Extract the latest user message and pass it as input to the agent
        user_input = st.session_state.messages[-1]["content"] if st.session_state.messages else ""

        # Ensure the input is a dictionary with the correct key
        response = agent_executor.invoke({"input": user_input}, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
        # st.write(get_search_results(vector_db, response))

