from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM with GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-1.5-flash",
                         google_api_key=os.getenv("GOOGLE_API_KEY"))


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

agent = initialize_agent(
    tools=[add_numbers_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    max_iterations=1,  
)

agent.run("first number is 10 second is 5")
















































# from langchain.agents import initialize_agent, AgentType
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.tools import tool



# # Initialize the LLM with GoogleGenerativeAI
# llm = GoogleGenerativeAI(model="gemini-1.5-flash",
#                          google_api_key="AIzaSyDIrIP-uFh7gZzjV_PUvHwOG6SE1xzpLuc")

# @tool
# def add_numbers_tool(input_data: str) -> str:
#     """ addition of two numbers. """
#     print("add_numbers_tool input_data",input_data)
#     try:
#         numbers = input_data.split(',')
#     except Exception as e:
#         return input_data
    
#     num1, num2 = int(numbers[0]), int(numbers[1])
#     result = num1 + num2
#     return f"The Sum of {num1} and {num2} is {result}"

# @tool
# def get_weather(city: str) -> str:
#     """ Get the weather of a city. """
#     print("get_weather input_data", city)
#     return f"The weather in {city} is clear with a temperature of 25°C."


# agent = initialize_agent(
#     tools=[add_numbers_tool,get_weather],
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     llm=llm,
#     verbose=True,
#     max_iterations=2,  
# )

# agent.run("2,3")