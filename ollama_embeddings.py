from langchain.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:5])

doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5])