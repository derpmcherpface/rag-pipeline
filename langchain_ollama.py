#!/usr/bin/python3
# requires: pip install langchain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

llm = Ollama(
    model="codellama", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
llm("Compute the value of Ackerman(1,2)")
