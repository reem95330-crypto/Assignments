"""Assignment 2 - Agent that answers questions using the top-3 web results.

Run:
  python agent.py "What is the difference between mitosis and meiosis?"

Env:
  OPENROUTER_API_KEY must be set.
Optional:
  OPENROUTER_MODEL (defaults to nvidia/nemotron-3-nano-30b-a3b:free)
"""

from __future__ import annotations

import os
import sys

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from prompts import SYSTEM_PROMPT
from tools import internet_search, fetch_url


def _chat_model() -> ChatOpenAI:
    """OpenRouter is OpenAI-compatible; langchain_openai can point at it via base_url."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_X_TITLE", "assignment-2-agent"),
        },
        temperature=0.2,
    )


def create_agent() -> AgentExecutor:
    llm = _chat_model()
    tools = [internet_search, fetch_url]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python agent.py "<your question>"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:]).strip()
    executor = create_agent()
    result = executor.invoke({"input": question})

    print("\n---\n")
    print(result.get("output", result))


if __name__ == "__main__":
    main()
