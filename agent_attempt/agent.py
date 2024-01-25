'''
Attempt to follow https://dev.to/timesurgelabs/how-to-make-an-ai-agent-in-10-minutes-with-langchain-3i2n
'''


# importing necessary variables
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
#from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

from langchain.tools import DuckDuckGoSearchRun
# loading environment variables
load_dotenv()

# setting up the duckduckgo search tool
#tool = DuckDuckGoSearchResults()

search = DuckDuckGoSearchRun()

