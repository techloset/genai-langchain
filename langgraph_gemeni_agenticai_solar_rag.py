from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable,RunnableLambda
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import uuid
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             google_api_key="AIzaSyAa9Jzw1m91loKmJvE1zU1vMm6k8QcClRw")


loader = WebBaseLoader(
    "https://aurorasolar.com/blog/how-do-solar-panels-work-everything-you-need-to-know/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(
    documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "Solar_panels_Knowledge",
    "Search for information about solar panel systems. For any questions solar system and it's working don't explain anything just to the point answers, you must use this tool!",
)



@tool
def compute_savings(monthly_cost: float) -> float:
    """
    Tool to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.

    Args:
        monthly_cost (float): The user's current monthly electricity cost.

    Returns:
        dict: A dictionary containing:
            - 'number_of_panels': The estimated number of solar panels required.
            - 'installation_cost': The estimated installation cost.
            - 'net_savings_10_years': The net savings over 10 years after installation costs.
    """
    def calculate_solar_savings(monthly_cost):
        # Assumptions for the calculation
        cost_per_kWh = 0.28
        cost_per_watt = 1.50
        sunlight_hours_per_day = 3.5
        panel_wattage = 350
        system_lifetime_years = 10

        # Monthly electricity consumption in kWh
        monthly_consumption_kWh = monthly_cost / cost_per_kWh

        # Required system size in kW
        daily_energy_production = monthly_consumption_kWh / 30
        system_size_kW = daily_energy_production / sunlight_hours_per_day

        # Number of panels and installation cost
        number_of_panels = system_size_kW * 1000 / panel_wattage
        installation_cost = system_size_kW * 1000 * cost_per_watt

        # Annual and net savings
        annual_savings = monthly_cost * 12
        total_savings_10_years = annual_savings * system_lifetime_years
        net_savings = total_savings_10_years - installation_cost

        return {
            "number_of_panels": round(number_of_panels),
            "installation_cost": round(installation_cost, 2),
            "net_savings_10_years": round(net_savings, 2)
        }

    # Return calculated solar savings
    return calculate_solar_savings(monthly_cost)


def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.

    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.

    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")

    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls

    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                # Format the error message for the user
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                # Associate the error message with the corresponding tool call ID
                tool_call_id=tc["id"],
            )
            # Iterate over each tool call to produce individual error messages
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.

    Args:
        tools (list): A list of tools to be included in the node.

    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        # Use a lambda function to wrap the error handler
        [RunnableLambda(handle_tool_error)],
        exception_key="error"  # Specify that this fallback is for handling errors
    )

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            You are a helpful customer support assistant specializing in Solar Panels.
            
            Your role is to provide users with information about solar panels and how they work by utilizing the Solar_panels_Knowledge tool. If the requested information isn't available, you may use the Tavily tool for an internet search.

            When users inquire about savings, you should ask for the following key detail:
            - Their current monthly electricity cost.

            If the user's message doesn't include this information or it's unclear, kindly request clarification. Avoid making any assumptions or guesses.
            
            Once you've gathered the necessary details, call the appropriate tool to assist the user.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)


# Define the tools the assistant will use
part_1_tools = [
    compute_savings,
    retriever_tool,
    TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY")),
]

# Bind the tools to the assistant's workflow
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    part_1_tools)



class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)

            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + \
                    [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break

        # Return the final state after processing the runnable
        return {"messages": result}





builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))


builder.add_edge(START, "assistant")  # Start with the assistant
builder.add_conditional_edges("assistant", tools_condition)
# Return to assistant after tool execution
builder.add_edge("tools", "assistant")


memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# import shutil

# Let's create an example conversation a user might have with the assistant
tutorial_questions = [
    "my montly cost is $100, what will i save"
]

config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

while True:
    data = graph.invoke(
        {"messages": ("user", input("How i can help you today?: "))}, config, stream_mode="values"
    )

    print(data)
