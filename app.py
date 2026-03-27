"""
Chainlit 웹 UI — 이 파일은 수정하지 않아도 됩니다.

agent.py의 create_base_agent()를 호출하여 에이전트를 생성하고,
사용자 메시지를 전달하여 응답을 표시합니다.

실행: uv run chainlit run app.py
"""

import logging
import uuid

# OpenTelemetry의 async 컨텍스트 충돌 경고 억제 (기능에 영향 없음)
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

import chainlit as cl
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler # 별칭

from agent import create_base_agent


@cl.on_chat_start
async def on_chat_start():
    """새 채팅 세션이 시작될 때 에이전트를 생성합니다."""
    agent = await create_base_agent()
    thread_id = str(uuid.uuid4())

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(content="안녕하세요! 무엇을 도와드릴까요?").send()


@cl.on_message
async def on_message(message: cl.Message):
    """사용자 메시지를 에이전트에 전달하고 응답을 스트리밍으로 표시합니다."""
    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")

    langfuse_handler = LangfuseCallbackHandler()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

    msg = cl.Message(content="")
    tool_steps: dict[str, cl.Step] = {}

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": message.content}]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            step = cl.Step(name=event["name"], type="tool")
            step.input = str(event["data"].get("input", ""))
            tool_steps[event["run_id"]] = step

        elif kind == "on_tool_end":
            step = tool_steps.pop(event["run_id"], None)
            if step:
                output = event["data"].get("output", "")
                step.output = output.content if hasattr(output, "content") else str(output)
                await step.send()

        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                await msg.stream_token(content)

    await msg.send()
