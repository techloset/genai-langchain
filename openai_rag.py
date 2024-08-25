from langchain_community.document_loaders import TextLoader
import os
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter

try:
    loader = TextLoader("data.txt")
    documents = loader.load()
except Exception as e:
    print("Error while loading file=", e)

# Create embeddings
embedding = OpenAIEmbeddings()

# Use a smaller chunk size to manage token limits
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Create the index with the specified embedding model and text splitter
index_creator = VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter
)
index = index_creator.from_loaders([loader])

# Specify the LLM for querying
llm = OpenAI(temperature=0)  # Replace with the correct LLM class initialization

# Query the index with the LLM
response = index.query("What is the name of your hotel?", llm=llm)
print(response)