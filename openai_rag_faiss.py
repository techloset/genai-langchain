import os
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data.txt")
documents = loader.load()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "hotel_information_sender",
    "Searches information about hotel from provided vector and return accurare as you can",
)
tools = [tool]

from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
chat_history = []
while True:
    user_input = input("You: ")
    if user_input=="quit":
        break
    chat_history.append(HumanMessage(content=user_input))

    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })

    print(f"Agent: ",response["output"])
    chat_history.append(AIMessage(content=response["output"]))