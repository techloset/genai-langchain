from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Example conversation with system messages
conversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Can you help me with my homework?"),
    AIMessage(content="Of course! What subject do you need help with?"),
    HumanMessage(content="I'm struggling with math."),
    AIMessage(content="I can help with that. What specific math problem are you working on?"),
]

conversation.append(HumanMessage(content="I'm struggling with math."))

for message in conversation:
    if isinstance(message, HumanMessage):
        print(f"User: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
    elif isinstance(message, SystemMessage):
        print(f"System: {message.content}")











# from langchain.schema import AIMessage, HumanMessage, SystemMessage

# # Example conversation with system messages
# conversation = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Can you help me with my homework?"),
#     AIMessage(content="Of course! What subject do you need help with?"),
#     HumanMessage(content="I'm struggling with math."),
#     AIMessage(content="I can help with that. What specific math problem are you working on?"),
# ]

# # Process conversation
# for message in conversation:
#     if isinstance(message, HumanMessage):
#         print(f"User: {message.content}")
#     elif isinstance(message, AIMessage):
#         print(f"AI: {message.content}")
#     elif isinstance(message, SystemMessage):
#         print(f"System: {message.content}")
