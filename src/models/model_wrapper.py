from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    ToolCall
)

def dict_to_lc_message(msg):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        # Check if tool_calls are present
        if "tool_calls" in msg:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    args=tc["args"]
                ) for tc in msg["tool_calls"]
            ]
            return AIMessage(content=content or "", tool_calls=tool_calls)
        else:
            return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content, tool_call_id=msg["tool_call_id"])
    else:
        raise ValueError(f"Unknown role: {role}")

llm = ChatOpenAI()

def wrapped_model(messages: list[dict[str, str]]):
    lc_messages = [dict_to_lc_message(m) for m in messages]
    response = llm.invoke(lc_messages)
    return {
        "role": "assistant",
        "content": response.content,
        # Optionally include tool_calls if any
        "tool_calls": [tc.dict() for tc in response.tool_calls] if response.tool_calls else None
    }
