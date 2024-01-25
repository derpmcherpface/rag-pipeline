
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
prompt = hub.pull("rlm/rag-prompt")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    },
)
print("loading document")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#print("creating vectorstore")
#vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(), persist_directory="./chroma_db")

# load vectorstore: 
embedding_function=OllamaEmbeddings()
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# query it
#query = "LLM Powered Autonomous Agents"
#docs = db.similarity_search(query)

# print results
#print(docs[0].page_content)

print("creating retriever")
retriever = db.as_retriever()


prompt = hub.pull("rlm/rag-prompt")
llm = Ollama(
    model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print('\n-what is task decomposition-\n')
rag_chain.invoke("What is Task Decomposition?")
print('\n-What is Tree of thoughts-\n')
rag_chain.invoke("What is Tree of thoughts?")
#print('\n-What are the types of memory for an ai agent-\n')
#rag_chain.invoke("What are the types of memory for an ai agent?\n")
print('\n-What are the types of Inner Product Search MIPS given in the context?-\n')
rag_chain.invoke("What are the types of Inner Product Search MIPS given in the context?")