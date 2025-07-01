"""
This module converts between AIMessage output formats, which are governed by the
``output_version`` attribute on ChatOpenAI. Supported values are ``"v0"``,
``"responses/v1"``, and ``"v1"``.

``"v0"`` corresponds to the format as of ChatOpenAI v0.3. For the Responses API, it
stores reasoning and tool outputs in AIMessage.additional_kwargs:

.. code-block:: python

    AIMessage(
        content=[
            {"type": "text", "text": "Hello, world!", "annotations": [{"type": "foo"}]}
        ],
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
            },
            "tool_outputs": [
                {"type": "web_search_call", "id": "websearch_123", "status": "completed"}
            ],
            "refusal": "I cannot assist with that.",
        },
        response_metadata={"id": "resp_123"},
        id="msg_123",
    )

``"responses/v1"`` is only applicable to the Responses API. It retains information
about response item sequencing and accommodates multiple reasoning items by
representing these items in the content sequence:

.. code-block:: python

    AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
                "id": "rs_123",
            },
            {
                "type": "text",
                "text": "Hello, world!",
                "annotations": [{"type": "foo"}],
                "id": "msg_123",
            },
            {"type": "refusal", "refusal": "I cannot assist with that."},
            {"type": "web_search_call", "id": "websearch_123", "status": "completed"},
        ],
        response_metadata={"id": "resp_123"},
        id="resp_123",
    )

There are other, small improvements as well-- e.g., we store message IDs on text
content blocks, rather than on the AIMessage.id, which now stores the response ID.

``"v1"`` represents LangChain's cross-provider standard format.

For backwards compatibility, this module provides functions to convert between the
formats. The functions are used internally by ChatOpenAI.
"""  # noqa: E501

import json
from typing import Union, cast

from langchain_core.messages import AIMessage, AIMessageChunk

_FUNCTION_CALL_IDS_MAP_KEY = "__openai_function_call_ids__"


def _convert_to_v03_ai_message(
    message: AIMessage, has_reasoning: bool = False
) -> AIMessage:
    """Mutate an AIMessage to the old-style v0.3 format."""
    if isinstance(message.content, list):
        new_content: list[Union[dict, str]] = []
        for block in message.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    # Store a reasoning item in additional_kwargs (overwriting as in
                    # v0.3)
                    _ = block.pop("index", None)
                    if has_reasoning:
                        _ = block.pop("id", None)
                        _ = block.pop("type", None)
                    message.additional_kwargs["reasoning"] = block
                elif block.get("type") in (
                    "web_search_call",
                    "file_search_call",
                    "computer_call",
                    "code_interpreter_call",
                    "mcp_call",
                    "mcp_list_tools",
                    "mcp_approval_request",
                    "image_generation_call",
                ):
                    # Store built-in tool calls in additional_kwargs
                    if "tool_outputs" not in message.additional_kwargs:
                        message.additional_kwargs["tool_outputs"] = []
                    message.additional_kwargs["tool_outputs"].append(block)
                elif block.get("type") == "function_call":
                    # Store function call item IDs in additional_kwargs, otherwise
                    # discard function call items.
                    if _FUNCTION_CALL_IDS_MAP_KEY not in message.additional_kwargs:
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {}
                    if (call_id := block.get("call_id")) and (
                        function_call_id := block.get("id")
                    ):
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][
                            call_id
                        ] = function_call_id
                elif (block.get("type") == "refusal") and (
                    refusal := block.get("refusal")
                ):
                    # Store a refusal item in additional_kwargs (overwriting as in
                    # v0.3)
                    message.additional_kwargs["refusal"] = refusal
                elif block.get("type") == "text":
                    # Store a message item ID on AIMessage.id
                    if "id" in block:
                        message.id = block["id"]
                    new_content.append({k: v for k, v in block.items() if k != "id"})
                elif (
                    set(block.keys()) == {"id", "index"}
                    and isinstance(block["id"], str)
                    and block["id"].startswith("msg_")
                ):
                    # Drop message IDs in streaming case
                    new_content.append({"index": block["index"]})
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        message.content = new_content
        if isinstance(message.id, str) and message.id.startswith("resp_"):
            message.id = None
    else:
        pass

    return message


def _convert_from_v03_ai_message(message: AIMessage) -> AIMessage:
    """Convert an old-style v0.3 AIMessage into the new content-block format."""
    # Only update ChatOpenAI v0.3 AIMessages
    # TODO: structure provenance into AIMessage
    is_chatopenai_v03 = (
        isinstance(message.content, list)
        and all(isinstance(b, dict) for b in message.content)
    ) and (
        any(
            item in message.additional_kwargs
            for item in [
                "reasoning",
                "tool_outputs",
                "refusal",
                _FUNCTION_CALL_IDS_MAP_KEY,
            ]
        )
        or (
            isinstance(message.id, str)
            and message.id.startswith("msg_")
            and (response_id := message.response_metadata.get("id"))
            and isinstance(response_id, str)
            and response_id.startswith("resp_")
        )
    )
    if not is_chatopenai_v03:
        return message

    content_order = [
        "reasoning",
        "code_interpreter_call",
        "mcp_call",
        "image_generation_call",
        "text",
        "refusal",
        "function_call",
        "computer_call",
        "mcp_list_tools",
        "mcp_approval_request",
        # N. B. "web_search_call" and "file_search_call" were not passed back in
        # in v0.3
    ]

    # Build a bucket for every known block type
    buckets: dict[str, list] = {key: [] for key in content_order}
    unknown_blocks = []

    # Reasoning
    if reasoning := message.additional_kwargs.get("reasoning"):
        buckets["reasoning"].append(reasoning)

    # Refusal
    if refusal := message.additional_kwargs.get("refusal"):
        buckets["refusal"].append({"type": "refusal", "refusal": refusal})

    # Text
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            block_copy = block.copy()
            if isinstance(message.id, str) and message.id.startswith("msg_"):
                block_copy["id"] = message.id
            buckets["text"].append(block_copy)
        else:
            unknown_blocks.append(block)

    # Function calls
    function_call_ids = message.additional_kwargs.get(_FUNCTION_CALL_IDS_MAP_KEY)
    for tool_call in message.tool_calls:
        function_call = {
            "type": "function_call",
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
            "call_id": tool_call["id"],
        }
        if function_call_ids is not None and (
            _id := function_call_ids.get(tool_call["id"])
        ):
            function_call["id"] = _id
        buckets["function_call"].append(function_call)

    # Tool outputs
    tool_outputs = message.additional_kwargs.get("tool_outputs", [])
    for block in tool_outputs:
        if isinstance(block, dict) and (key := block.get("type")) and key in buckets:
            buckets[key].append(block)
        else:
            unknown_blocks.append(block)

    # Re-assemble the content list in the canonical order
    new_content = []
    for key in content_order:
        new_content.extend(buckets[key])
    new_content.extend(unknown_blocks)

    new_additional_kwargs = dict(message.additional_kwargs)
    new_additional_kwargs.pop("reasoning", None)
    new_additional_kwargs.pop("refusal", None)
    new_additional_kwargs.pop("tool_outputs", None)

    if "id" in message.response_metadata:
        new_id = message.response_metadata["id"]
    else:
        new_id = message.id

    return message.model_copy(
        update={
            "content": new_content,
            "additional_kwargs": new_additional_kwargs,
            "id": new_id,
        },
        deep=False,
    )


def _convert_to_v1_from_chat_completions(message: AIMessage) -> AIMessage:
    """Mutate a Chat Completions message to the v1 format."""
    if isinstance(message.content, str):
        if message.content:
            message.content = [{"type": "text", "text": message.content}]
        else:
            message.content = []

    for tool_call in message.tool_calls:
        if id_ := tool_call.get("id"):
            message.content.append({"type": "tool_call", "id": id_})

    if "tool_calls" in message.additional_kwargs:
        _ = message.additional_kwargs.pop("tool_calls")

    if "token_usage" in message.response_metadata:
        _ = message.response_metadata.pop("token_usage")

    return message


def _convert_to_v1_from_chat_completions_chunk(chunk: AIMessageChunk) -> AIMessageChunk:
    result = _convert_to_v1_from_chat_completions(cast(AIMessage, chunk))
    return cast(AIMessageChunk, result)


def _convert_from_v1_to_chat_completions(message: AIMessage) -> AIMessage:
    """Convert a v1 message to the Chat Completions format."""
    # TODO: currently unused, will this break non-OpenAI providers?
    if isinstance(message.content, list):
        new_content = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                # Strip annotations
                new_content.append({"type": "text", "text": block["text"]})
            else:
                pass
        return message.model_copy(update={"content": new_content})

    return message
