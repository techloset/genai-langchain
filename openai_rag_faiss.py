
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


loader = TextLoader("data.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()


tool = create_retriever_tool(
    retriever,
    "hotel_information_sender",
    "Searches information about hotel from provided vector and return accurare as you can",
)
tools = [tool]



prompt = hub.pull("hwchase17/openai-tools-agent")



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