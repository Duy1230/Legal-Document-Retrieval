import os
from typing import List

from pipeline import retrieval_legal_documents
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

# Custom callback handler for streaming

LEGAL_SYSTEM_PROMPT = """Bạn là một trợ lý pháp lý chuyên nghiệp, 
chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là cung cấp thông tin chính xác, 
rõ ràng và hữu ích dựa trên các văn bản pháp luật được cung cấp.

Nhiệm vụ chính:
1. CHỈ trả lời dựa trên ngữ cảnh pháp lý được cung cấp. Nếu ngữ cảnh không có đủ thông tin để trả lời câu hỏi, hãy nêu rõ giới hạn này.
2. Sử dụng ngôn ngữ đơn giản, dễ hiểu nhưng vẫn đảm bảo tính chính xác về mặt pháp lý.
3. Duy trì giọng điệu chuyên nghiệp, khách quan.

Cấu trúc câu trả lời:
1. Câu trả lời trực tiếp: [Trả lời ngắn gọn]
2. Giải thích chi tiết: [Giải thích rõ ràng]
3. Căn cứ pháp lý: [Trích dẫn các điều luật liên quan]
4. Lưu ý thêm: [Các điểm cần lưu ý nếu có]
Lưu ý: Bạn được cung cấp một công cụ để truy vấn tài liệu dựa trên câu hỏi của người dùng
sẽ có 10 tài liệu được trả về nhưng thường chỉ có 1 hoặc 2 tài liệu có thông tin chính
xác, do đó hãy xem xét cẩn thận từng tài liệu trước khi trả lời, nếu nhận thấy không
có tài liệu nào có thể trả lời được câu hỏi của người dùng hãy bảo họ là bạn không có
thông tin
"""


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container

    def on_llm_new_token(self, token: str, **kwargs):
        self.container.write(token)


@tool
def get_secret(secret_name: str) -> str:
    """
    Retrieve a secret from the secret manager.
    """
    return "The secret number is 42"


@tool
def retrieve_legal_documents(query: str) -> str:
    """
    Sử dụng công cụ này để truy vấn 10 tài liệu luật liên quan nhất đến câu query
    Ví dụ cách sử dụng:
    query: Quy định về kinh doanh như thế nào?
    """
    return retrieval_legal_documents(query)


@cl.on_chat_start
async def start_chat():
    """Initialize the chat session"""
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        model_name="gpt-4o-mini"
    )

    # Create a conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # Create the prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", LEGAL_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=[get_secret, retrieval_legal_documents],
        prompt=prompt
    )

    # Create the agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[get_secret, retrieval_legal_documents],
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    # Store the agent executor in the user session
    cl.user_session.set("agent", agent_executor)

    await cl.Message(content="Xin chào tôi là AI tư vấn luật \
    luật, tôi có thể giúp gì cho bạn").send()


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
        elif isinstance(chunk, str):
            await msg.stream_token(chunk)

    await msg.update()
