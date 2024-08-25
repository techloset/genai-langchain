# How to get started with langchain
- Python based dev container with using vs code remote explorer

- Poetry install https://python-poetry.org/docs/#installing-with-the-official-installer

``` curl -sSL https://install.python-poetry.org | python3 - ```

- Lanchain install https://python.langchain.com/v0.1/docs/get_started/installation/

```
  poetry add langchain-core
  poetry add langchain-community
  poetry add langchain-openai
```
- Create main.py file
- Get open ai api key https://platform.openai.com/api-keys
- Run   poetry add python-dotenv

## Core Langchain Concepts: 
- from langchain_core.messages import AIMessage,HumanMessage, ToolMessage,BaseMessage
- from langchain_core.tools import tool
- from langchain_core.prompts import ChatPromptTemplate
- from langchain_core.runnables import Runnable, RunnableConfig
- from langchain_core.utils.function_calling import convert_to_openai_function
- from langgraph.prebuilt import ToolNode
- from langgraph.graph import END, StateGraph, START
- from langgraph.prebuilt import tools_condition
- from langgraph.graph.message import AnyMessage, add_messages
- from langgraph.graph import MessagesState, StateGraph, START
- from langchain.agents import Tool
- from langchain.utilities import GoogleSerperAPIWrapper
- from langchain.chains import APIChain
- from langchain.vectorstores import FAISS
- from langchain.embeddings import OpenAIEmbeddings
- from langchain.callbacks.base import AsyncCallbackHandler
- from langchain.callbacks.manager import AsyncCallbackManager
- from langchain.chains.conversation.memory import ConversationBufferMemory
- from langchain.memory import ConversationBufferWindowMemory
- from langchain.text_splitter import CharacterTextSplitter
- from langchain.docstore.document import Document
- from langchain.vectorstores import FAISS
- from langchain.memory import ConversationBufferMemory



  
