from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import sys


llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyDIrIP-uFh7gZzjV_PUvHwOG6SE1xzpLuc")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "How much is 2+2?"
# print(chain.invoke({"question": question}))

for chunk in llm.stream("Tell me a short poem about snow"):
    sys.stdout.write(chunk)
    sys.stdout.flush()

# from langchain_community.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()

# try:
#     loader=TextLoader("Hotel_Alexandra.txt")
#     print("loader",loader)
# except Exception as e:
#     print("Error while loading file=",e)
# index=VectorstoreIndexCreator().from_loaders([loader])
# print("index",index)

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

# existing_response = index.query("What is the location of the hotel?")
# print("existing_response",existing_response)

# from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import SimpleSequentialChain, LLMChain
# from dotenv import load_dotenv

# import os
# load_dotenv()

# llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"))

# llm2 = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     huggingfacehub_api_token= os.getenv("HUGGINGFACE_TOKEN")
# )



# prompt0 = PromptTemplate(template="Context: of this chat", inputVriable=["chat"])



# prompt1 = PromptTemplate(template="Give me the famous place of {Text}", inputVriable=["text"])

# prompt2 = PromptTemplate(template="Wha is the location of place {text}", inputVriable=["text"])



# chain1 = LLMChain(llm=llm, prompt=prompt1)
# chain2 = LLMChain(llm=llm, prompt=prompt2)

# chain = SimpleSequentialChain(chains=[chain1, chain2])

# result = chain.invoke("Faisalabad")

# print(result)




# from langchain_huggingface import HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta"
#     huggingfacehub_api_token= "sdf"
# )