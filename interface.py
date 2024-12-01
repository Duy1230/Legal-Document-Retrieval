import os
from typing import List

import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = ""

# Custom callback handler for streaming


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container

    def on_llm_new_token(self, token: str, **kwargs):
        self.container.write(token)


@tool
def get_secret(secret_name: str) -> str:
    """
    Using this when user ask what is the secret number
    """
    return "The secret number is 42"


@cl.on_chat_start
async def start_chat():
    """Initialize the chat session"""
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        model_name="gpt-3.5-turbo"
    )

    # Create tools
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Add the custom get_secret tool
    tools = [wiki_tool, get_secret]

    # Create a conversation memory
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="output"
    )

    # Define the agent prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with access to Wikipedia search. Use the tool to find information when needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ])

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # Store the agent executor in the user session
    cl.user_session.set("agent", agent_executor)

    # Send initial message
    await cl.Message(
        content="Hello! I'm an AI assistant with Wikipedia search and secret retrieval capabilities. Ask me anything!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""

    # Get the agent executor from user session
    agent_executor = cl.user_session.get("agent")

    # Create a message container
    msg = cl.Message(content="")
    await msg.send()

    # Run the agent with streaming
    async for chunk in agent_executor.astream(
        {"input": message.content},
    ):
        # Check if 'output' is in the chunk to handle different streaming behaviors
        if 'output' in chunk:
            await msg.stream_token(chunk["output"])
        elif 'intermediate_steps' in chunk:
            # Optionally handle intermediate steps if needed
            pass

    await msg.update()

if __name__ == "__main__":
    # Run the Chainlit app
    cl.run()
