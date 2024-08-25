from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate


from dotenv import load_dotenv
import os

load_dotenv()


llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt = PromptTemplate(template="", inputVriable=["context"])


chain = prompt | llm
while True:
    human_message = input("Enter your story context:")
    ai_message = chain.invoke({"context": human_message})
    print(ai_message)


















# from langchain_google_genai import GoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os
# load_dotenv()


# llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# prompt = PromptTemplate(template="Create the story about two friends who are going to the market to buy some fruits?. response by characters {characters}", inputVriable=["characters"])

# chain = prompt | llm

# response = chain.invoke({"characters": "Ali and Naveed"})

# print(response)

