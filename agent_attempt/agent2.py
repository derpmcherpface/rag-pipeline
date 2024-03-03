'''
Sequel to agent.py
Hopefully somewhat simplified
'''


# importing necessary variables
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

from langchain.tools import DuckDuckGoSearchRun

# -----------------------------
#ollama stuff
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
#--------------------------------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

## Here's where we diverge from the previous example.
llm = Ollama(
    model="mistral:instruct", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    #model="vicuna", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
agent = initialize_agent(
    tools=[tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

'''
agent = initialize_agent(
    tools=[tool],
    agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)'''


  #  tools=tools,
prompt="Describe what you know about Napoleon Bonaparte."
print(agent.run(prompt))

# https://smith.langchain.com/hub/hwchase17/react
'''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}


'''


# look into: https://python.langchain.com/docs/modules/agents/how_to/handle_parsing_errors