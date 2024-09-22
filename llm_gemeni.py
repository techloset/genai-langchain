from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGoogleGenerativeAI( model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are fruits assistent, you have to act like answer about fruits qustions"),
        SystemMessage(content="Please don't explain anything else, just answer the question related to fruits"),
         SystemMessage(content="Don't respond if user ask anything other than fruits related questions, always respond in json, your return should be this, this"),
    ]
)


while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    print(user_input)
    prompt_template.append(HumanMessage(content=user_input))
    prompt = prompt_template.format()
    print("Prompt: ",prompt)
    response = llm.invoke(prompt)
    print("LLM Response",response)
    prompt_template.append(AIMessage(content=response.content))
    





































# from langchain_google_genai import GoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from dotenv import load_dotenv
# import os
# load_dotenv()

# llm = GoogleGenerativeAI(
#     model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))



# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content="You are fruits assistent, you have to act like answer about fruits qustions"),
#     ]
# )



# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     print(user_input)
#     prompt_template.append(HumanMessage(content=user_input))
#     prompt = prompt_template.format()
#     print("Prompt: ",prompt)
#     response = llm.invoke(user_input)
#     print("Final==>>",response)
    # prompt_template.append(AIMessage(content=response))
    # print("Final==>>",response)
    
    
    
    
    
    
    


# print(response)
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# from dotenv import load_dotenv
# import os
# load_dotenv()


# llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
# memory = ConversationBufferMemory()
# chain = ConversationChain(llm=llm,memory=memory)
# prompt = "My Name is Naveed and I am a software engineer. I am working on a project and I need some help with it. Can you help me with it?"
# response = chain.invoke(prompt)
# response2 = chain.invoke("what is my Name?")
# chain.invoke("i do own my company named as techloset solutions")
# chain.invoke("i teach programming languages in piaic")

# final = chain.invoke("what is my company name?, don't explain anything else")

# # print(response)
# print("Final==>>",final)
