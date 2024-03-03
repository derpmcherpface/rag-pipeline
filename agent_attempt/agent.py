'''
Attempt to follow https://dev.to/timesurgelabs/how-to-make-an-ai-agent-in-10-minutes-with-langchain-3i2n
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



# loading environment variables
load_dotenv()

# setting up the duckduckgo search tool
ddg_search = DuckDuckGoSearchResults()

#Defining Headers for Web Requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

# Parsing HTML Content
def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

# Creating the web fetcher tool
web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

# Setting up the summarizer
prompt_template = "Summarize the following content: {content}"

## Here's where we diverge from the previous example.
llm = Ollama(
    model="vicuna", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)


tools = [ddg_search, web_fetch_tool, summarize_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

# here it is stupid and tries to display using the web fetcher triggering an error. Find better prompt or model
prompt = "Research how to use the requests library in Python. Use your tools to search and summarize content into a guide on how to use the requests library."

print(agent.run(prompt))





#search = DuckDuckGoSearchRun()

