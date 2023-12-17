



import bs4
import chromadb
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import Ollama

from langchain.embeddings import OllamaEmbeddings

from langchain.document_loaders import WebBaseLoader

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#query = "What did the president say about Ketanji Brown Jackson"

query="The heuristic function determines when the trajectory"

embedding_function=OllamaEmbeddings()
# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)


