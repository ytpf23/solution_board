from openai import OpenAI
import streamlit as st
from typing import List, Tuple, Dict

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from langchain.agents import (
    initialize_agent,
    AgentType,
    create_openai_tools_agent,
    AgentExecutor,
)
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

import os

from langchain.globals import set_debug, set_verbose

set_verbose(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, which company do you come from?"}
    ]

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}


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


hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


vector_db = PGVector(
    collection_name=params.collection_name,
    connection=params.psql_conn_string,
    embeddings=hf_embeddings,
)


def create_research_agent(vector_db):
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    @tool
    def paper_search(query: str) -> List[dict]:
        """
        Search scientific papers database with the refined query.
        Only use after gathering user background information.
        """
        results = vector_db.similarity_search_with_score(query, k=5)
        papers = [
            {
                "title": doc.metadata["title"],
                "email": doc.metadata["email"],
                "abstract": doc.page_content,
                "citations": doc.metadata["n_citation"],
                "year": doc.metadata["year"],
                "relevance_score": float(score),
            }
            for doc, score in results
        ]
        # Update session state with retrieved papers
        st.session_state.retrieved_papers = papers
        return papers

    @tool
    def analyze_business_impact() -> str:
        """
        Generate specific business implementation strategy and impact analysis based on the retrieved papers.
        Start with one exiting one sentence summary of business application that business user can understand.
        After that provide the paper real name in the breakets.

        Enchance you responce with 1-2 sentence explanation of business value of implementation. what it can improve and how exactly
        Make your response structure concise and to the point.
        Use bulletpoints and section identifiers.
        ALWYS ADD Business Value section in all responces.

        """
        return f"Business impact analysis and implementation strategy for the retrevied papers."

    system_prompt = """You are SciCombinator, an AI research consultant specializing in
                        connecting business problems with scientific solutions.
                        Review the chat history before responding to avoid asking the same questions.
                        Follow this exact process:
                        1. BEFORE searching:
                        - Based on the user company description analyze the user's industry and problem from chat history
                        - Consider industry-specific terminology that can be commonly used on scientific papers but user might not be aware of
                        - Formulate a specific, technical search query that will find relevant papers and Use paper_search() with your refined query to find relevant papers
                        2. For each relevant paper:
                        - Start with a business applications relevant for the user and add in breakets paper title. Never start with papers title. Main idea is to translate scientific language to business language.
                        - Use analyze_business_impact() to create specific implementation strategies
                        - Focus on practical applications
                        - Highlight potential ROI and competitive advantages
                        3. Summarize your recommendations in a clear, business-friendly manner
                        Always maintain a professional, consultative tone and ensure recommendations are
                        practical and actionable. If you need clarification, ask follow-up questions.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # st.write(prompt)

    tools = [paper_search, analyze_business_impact]
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", temperature=1, openai_api_key=OPENAI_API_KEY
    )
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,  # Add memory to the executor
        verbose=True,
    )


agent_executor = create_research_agent(vector_db)


def main():
    st.title("SciCombinator")
    st.caption(
        "Connecting industry problems to scientific solutions powered by Artificial Intelligence"
    )
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieved_papers" not in st.session_state:
        st.session_state.retrieved_papers = []
    if "sources_container" not in st.session_state:
        st.session_state.sources_container = st.empty()
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # Chat input
    if prompt := st.chat_input("Describe your business challenge..."):
        # Clear previous sources container
        st.session_state.sources_container.empty()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if not OPENAI_API_KEY:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["output"]}
            )
            st.write(response["output"])
            # Update sources in the container
            st.session_state.sources_container = st.empty()
            with st.session_state.sources_container.container():
                with st.expander(" --- Retrieved Sources --- ", expanded=False):
                    if len(st.session_state.retrieved_papers) > 0:
                        df = pd.DataFrame(st.session_state.retrieved_papers)
                        df["abstract"] = df["abstract"].str[:200] + "..."
                        df["relevance_score"] = df["relevance_score"].round(3)
                        df = df[
                            [
                                "title",
                                "email",
                                "year",
                                "citations",
                                "relevance_score",
                                "abstract",
                            ]
                        ].rename(
                            columns={
                                "citations": "Citation Count",
                                "email": "Key Contact",
                                "relevance_score": "Relevance Score",
                                "year": "Year",
                                "title": "Title",
                                "abstract": "Abstract Preview",
                            }
                        )
                        st.dataframe(
                            df,
                            column_config={
                                "Title": st.column_config.TextColumn(
                                    "Title", width="medium"
                                ),
                                "Year": st.column_config.NumberColumn(
                                    "Year", format="%d"
                                ),
                                "Citation Count": st.column_config.NumberColumn(
                                    "Citations"
                                ),
                                "Relevance Score": st.column_config.NumberColumn(
                                    "Relevance",
                                    help="Higher score indicates more relevance to query",
                                    format="%.3f",
                                ),
                                "Abstract Preview": st.column_config.TextColumn(
                                    "Abstract Preview", width="large"
                                ),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("No papers retrieved yet.")


if __name__ == "__main__":
    main()
