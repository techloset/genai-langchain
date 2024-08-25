from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Initialize memory
memory = ConversationBufferMemory()

# Start the conversation with a system message
memory.save_context({}, {"content": "You are a customer support assistant for an online retail store."})

# Simulate a conversation
memory.save_context({"input": "Hello, do you have the new iPhone in stock?"}, 
                   {"output": "Yes, the new iPhone is currently in stock. Would you like to place an order?"})

memory.save_context({"input": "Can you check the status of my order #12345?"}, 
                   {"output": "Your order #12345 has been shipped and is expected to arrive in 3 days."})

memory.save_context({"input": "What's your return policy?"}, 
                   {"output": "Our return policy allows you to return items within 30 days of purchase. Would you like to initiate a return?"})

# Later, the AI might use this memory to respond more intelligently
for message in memory.load_memory():
    if isinstance(message, HumanMessage):
        print(f"User: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
    elif isinstance(message, SystemMessage):
        print(f"System: {message.content}")
