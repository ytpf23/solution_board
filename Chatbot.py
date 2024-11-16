from openai import OpenAI
import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
import os 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("TechCombinator")
st.caption("Conncecting industry problems to scientific solutions powered by the Artificial Intelligence")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, which company do you come from?"}]

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
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

