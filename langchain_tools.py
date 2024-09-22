from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
import request
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template = PromptTemplate(
    input_variables=["input"],
    template="You are a tool caller, you have to call the the tool named add_numbers_tool in case if there is any addition required, please don't send any explanation while calling the function, just send the two numbers what users provided e.g 5,2, even though user give the the sentance you have to find two numbers and pass to the functions, user input is: {input}\n")


@tool
def get_weather_tool(city: str) -> str:
    """ Get the weather of a city """
    print("get_weather_tool input_data",city)
    output = request(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=API_KEY")    
    return output

@tool
def add_numbers_tool(input_data: str) -> str:
    """ addition of two numbers. """
    print("add_numbers_tool input_data",input_data)
    try:
        numbers = input_data.split(',')
    except Exception as e:
        return "No numbers found"
    num1, num2 = int(numbers[0]), int(numbers[1])   
    result = num1 + num2
    return f"The Sum of {num1} and {num2} is {result}"

@tool
def multiply_tool(input_data: str) -> str:
    """ Multipy of two numbers """
    print("multiply_tool input_data",input_data)    
    return "Multiplication is 15"
    

chain = RunnableSequence(
    prompt_template, 
    llm,
    get_weather_tool
    )

output = chain.invoke("my first number is 10 and second should be minus of first number by 2")
print("output",output)
    
























































# from langchain_google_genai import GoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain_core.tools import tool
# from langchain_core.runnables import RunnableSequence


# llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyDIrIP-uFh7gZzjV_PUvHwOG6SE1xzpLuc")

# prompt_template = PromptTemplate(
#     input_variables=["input"],
#     template="You are a tool caller. You have to find two numbers from the users input, call the Addition Tool to compute their sum, only pass the numbers by comma seperated don't add anything in the input of tool, please don't pass anything in the perametter of tool, just pass the string with numbers e.g 10,5 don't return any of eglish words of explanation just the numbres what you find don't explain or add any words.\nNumbers: {input}\n"
# )


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

# chain = RunnableSequence(
#     prompt_template,  # First, the LLM will format the input with the prompt template
#     llm,
#     add_numbers_tool,
#     )

# # Test the chain with two numbers
# input_numbers = "my first number is 10 and second should be minus of first number by 2"
# output = chain.invoke(input_numbers)

# print(output)