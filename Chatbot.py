from openai import OpenAI
import streamlit as st
from typing import List, Tuple, Dict

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
from langchain.memory import ConversationBufferMemory

import os 

from langchain.globals import set_debug, set_verbose
set_verbose(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


vector_db = PGVector(
    collection_name=params.collection_name,
    connection=params.psql_conn_string,
    embeddings=hf_embeddings
)

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     if not OPENAI_API_KEY:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()

#     llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, streaming=True)
#     search = DuckDuckGoSearchRun(name="Search")
#     search_agent = initialize_agent(
#         [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
#     )

#     with st.chat_message("assistant"):
#         st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
#         # response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
#         # Extract the latest user message and pass it as input to the agent
#         user_input = st.session_state.messages[-1]["content"] if st.session_state.messages else ""

#         # Ensure the input is a dictionary with the correct key
#         response = agent_executor.invoke({"input": user_input}, callbacks=[st_cb])
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.write(response)
#         # st.write(get_search_results(vector_db, response))

def create_research_agent(vector_db):
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    @tool
    def paper_search(query: str) -> List[dict]:
        """
        Search scientific papers database with the refined query.
        Only use after gathering user background information.
        """
        results = vector_db.similarity_search_with_score(query, k=5)
        return [{
            'title': doc.metadata['title'],
            'abstract': doc.page_content,
            'citations': doc.metadata['n_citation'],
            'year': doc.metadata['year']
        } for doc, score in results]
    @tool
    def analyze_business_impact(company_info: Dict, paper_details: Dict) -> str:
        """
        Generate specific business implementation strategy and impact analysis.
        Args:
            company_info: Dict containing industry, company details, and problem
            paper_details: Scientific paper information
        """
        return f"Business impact analysis for {company_info['company_name']}"
    @tool
    def gather_user_context() -> Dict:
        """
        Gather essential information about the user's business context.
        Must ask about:
        1. Industry sector
        2. Specific business problem
        3. Existing solutions or current approaches tried
        """
        return {
            "questions": [
                "What industry sector does your company operate in?",
                "Can you describe the specific business problem you're trying to solve?",
                "What approaches have you already tried to solve this problem?",
                "What existing solutions are there?"
            ]
        }
    system_prompt = """You are Tech Combinator, an AI research consultant specializing in
    connecting business problems with scientific solutions.
    Review the chat history before responding to avoid asking the same questions.
    Follow this exact process:
    1. START by using gather_user_context() to ask important questions about the user's
       business context. Wait for user responses. Use this gather_user_context() tool maximal once at the beginning of conversation.
       ALWAYS Skip this step if the information is already in chat history.
    2. AFTER getting context, BEFORE searching:
       - Analyze the user's industry and problem from chat history
       - Consider industry-specific terminology that can be commonly used on scientific papers but user might not be aware of
       - Formulate a specific, technical search query that will find relevant papers and Use paper_search() with your refined query to find relevant papers
    3. For each relevant paper:
       - Use analyze_business_impact() to create specific implementation strategies
       - Focus on practical applications
       - Highlight potential ROI and competitive advantages
    4. Summarize your recommendations in a clear, business-friendly manner
    Always maintain a professional, consultative tone and ensure recommendations are
    practical and actionable. If you need clarification, ask follow-up questions.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    st.write(prompt)

    tools = [gather_user_context, paper_search, analyze_business_impact]
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=1,
        openai_api_key=OPENAI_API_KEY
    )
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,  # Add memory to the executor
        verbose=True
    )

agent_executor = create_research_agent(vector_db)

# Usage with Streamlit
def main():

    st.title("TechCombinator")
    st.caption("Conncecting industry problems to scientific solutions powered by the Artificial Intelligence")
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # Chat input
    if prompt := st.chat_input("Describe your business challenge..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if not OPENAI_API_KEY:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent_executor.invoke(
                {"input": prompt},
                callbacks=[st_cb]
            )
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            st.write(response["output"])

if __name__ == "__main__":
    main()

