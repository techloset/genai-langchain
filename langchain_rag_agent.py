from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM with GoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                         google_api_key=os.getenv("GOOGLE_API_KEY"))

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

loader = WebBaseLoader("https://www.techloset.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "techloset_search",
    "Search for information about Techloset. For any questions about Techloset Solutions, you must use this tool!",
)


@tool
def add_numbers_tool(input_data: str) -> str:
    """ addition of two numbers. """
    print("add_numbers_tool input_data",input_data)
    try:
        numbers = input_data.split(',')
    except Exception as e:
        return input_data
    
    num1, num2 = int(numbers[0]), int(numbers[1])
    result = num1 + num2
    return f"The Sum of {num1} and {num2} is {result}"

tools = [search, retriever_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    agent_with_chat_history.invoke(
        {"input": input("How can I help you today? : ")},
        config={"configurable": {"session_id": "test123"}},
    )



