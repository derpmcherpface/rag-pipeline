# the tutorial: https://www.pinecone.io/learn/series/langchain/langchain-intro/
from langchain import PromptTemplate

import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

print("importing langchain")
from langchain import HuggingFaceHub, LLMChain


#repo_id=google/flan-t5-xl

print("initializing LLM")
# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='databricks/dolly-v2-3b',
    model_kwargs={"temperature":0, "max_length":64}
)

print("creating LLM chain")
# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

print("running LLM chain")
# ask the user question about NFL 2010
print(llm_chain.run(question))

