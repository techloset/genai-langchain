from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)

chain = ConversationChain(llm=llm, memory=memory)

while True: 
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = chain.invoke(user_input)
    print("Final==>>",response)



























# from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory,ConversationSummaryMemory, ConversationSummaryBufferMemory
# from langchain.chains import ConversationChain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


# memory = ConversationBufferMemory()


# chain = ConversationChain(llm=llm, memory=memory)

# while True: 
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     response = chain.invoke(user_input, user="123")
#     print("Final==>>",response)






