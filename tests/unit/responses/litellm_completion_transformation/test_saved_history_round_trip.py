"""
Round trip a Responses API client (e.g. the OpenAI Agents SDK) runs through the bridge:
stream a turn, run its tools, save and reload the history as JSON, send it on the next turn.
"""

import json
from typing import Final

import httpx
import respx

import litellm

SIGNATURE: Final = "EqQBCkYIBRgCKkB-anthropic-signature"
THINKING: Final = "Two cities, so two weather calls."
TOOLS: Final = [
    {
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]
USER_TURN: Final = {"role": "user", "content": "Weather in Paris and Tokyo?"}
CITY_WEATHER: Final = {"Paris": "22C", "Tokyo": "31C"}


def _message_start(message_id: str) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    }


def _tool_use_events(index: int, tool_use_id: str, city: str) -> list[dict[str, object]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": tool_use_id, "name": "get_weather", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps({"city": city})},
        },
        {"type": "content_block_stop", "index": index},
    ]


CLAUDE_TOOL_TURN: Final = [
    _message_start("msg_tools"),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": THINKING}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": SIGNATURE}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Checking both."}},
    {"type": "content_block_stop", "index": 1},
    *_tool_use_events(2, "toolu_paris", "Paris"),
    *_tool_use_events(3, "toolu_tokyo", "Tokyo"),
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 50}},
    {"type": "message_stop"},
]
CLAUDE_ANSWER_TURN: Final = [
    _message_start("msg_answer"),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Paris 22C, Tokyo 31C."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
]
GEMINI_ANSWER: Final = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "Paris 22C, Tokyo 31C."}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 9, "totalTokenCount": 19},
}


def _sse(events: list[dict[str, object]]) -> httpx.Response:
    body: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _gemini_sse(chunk: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200, content=f"data: {json.dumps(chunk)}\n\n".encode(), headers={"content-type": "text/event-stream"}
    )


def _stream_turn(
    model: str, items: list[object], reply: httpx.Response
) -> tuple[dict[str, object], list[dict[str, object]]]:
    with respx.mock() as provider:
        route: Final = provider.route().mock(return_value=reply)
        stream: Final = litellm.responses(
            model=model, input=items, tools=TOOLS, reasoning={"effort": "low"}, stream=True, api_key="test-key"
        )
        completed: Final = [event.response for event in stream if event.type == "response.completed"]
        request: Final = json.loads(route.calls.last.request.content)
    return request, [item.model_dump(exclude_none=True) for item in completed[-1].output]


def _saved_history_after_tool_turn() -> list[object]:
    _, output = _stream_turn("anthropic/claude-sonnet-5-5", [USER_TURN], _sse(CLAUDE_TOOL_TURN))
    tool_outputs: Final = [
        {
            "type": "function_call_output",
            "call_id": item["call_id"],
            "output": CITY_WEATHER[json.loads(item["arguments"])["city"]],
        }
        for item in output
        if item["type"] == "function_call"
    ]
    return json.loads(json.dumps([USER_TURN, *output, *tool_outputs]))


def test_streamed_tool_turn_saves_one_distinct_item_per_output() -> None:
    output: Final = [item for item in _saved_history_after_tool_turn()[1:] if item["type"] != "function_call_output"]

    assert [item["type"] for item in output] == ["reasoning", "message", "function_call", "function_call"]
    assert len({item["id"] for item in output}) == len(output)


def test_saved_claude_history_replays_to_claude_with_signature_and_tool_order() -> None:
    history: Final = _saved_history_after_tool_turn()

    request, _ = _stream_turn("anthropic/claude-sonnet-5-5", history, _sse(CLAUDE_ANSWER_TURN))

    assistant, tool_results = request["messages"][1], request["messages"][2]
    assert assistant["content"][0] == {"type": "thinking", "thinking": THINKING, "signature": SIGNATURE}
    assert [block["id"] for block in assistant["content"] if block["type"] == "tool_use"] == [
        "toolu_paris",
        "toolu_tokyo",
    ]
    assert [block["tool_use_id"] for block in tool_results["content"]] == ["toolu_paris", "toolu_tokyo"]


def test_saved_claude_thinking_reaches_gemini_once_as_unsigned_thought() -> None:
    history: Final = _saved_history_after_tool_turn()

    request, _ = _stream_turn("gemini/gemini-3.5-flash", history, _gemini_sse(GEMINI_ANSWER))

    model_parts: Final = next(content["parts"] for content in request["contents"] if content["role"] == "model")
    assert [part for part in model_parts if part.get("text") == THINKING] == [{"thought": True, "text": THINKING}]
    assert SIGNATURE not in json.dumps(request)
