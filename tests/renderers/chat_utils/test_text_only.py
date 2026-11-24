# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    parse_chat_messages,
    parse_chat_messages_async,
    unsupported_attachment_note,
)
from vllm.renderers.chat_utils import chat_message as chat_message_module

MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


@pytest.fixture(scope="function")
def mistral_model_config():
    return ModelConfig(
        MISTRAL_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={
            "image": 2,
        },
    )


def test_parse_chat_messages_empty_system(
    mistral_model_config,
):
    # Test string format
    conversation, _, _ = parse_chat_messages(
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Who are you?"}],
            },
        ],
        mistral_model_config,
        content_format="string",
    )
    assert conversation == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Who are you?"},
    ]

    # Test openai format
    conversation, _, _ = parse_chat_messages(
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Who are you?"}],
            },
        ],
        mistral_model_config,
        content_format="openai",
    )
    assert conversation == [
        {"role": "system", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
    ]


@pytest.mark.asyncio
async def test_text_only_chat_does_not_initialize_media_connector(
    mistral_model_config,
    monkeypatch,
):
    load_connector = MagicMock()
    monkeypatch.setattr(MEDIA_CONNECTOR_REGISTRY, "load", load_connector)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Who are you?"}
    ]

    parse_chat_messages(
        messages,
        mistral_model_config,
        content_format="string",
    )
    await parse_chat_messages_async(
        messages,
        mistral_model_config,
        content_format="string",
    )

    load_connector.assert_not_called()


def test_parse_chat_messages_include_thinking_chunk(mistral_model_config):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "thinking",
                    "closed": True,
                    "thinking": "Only return the answer when you are confident.",
                },
            ],
        },
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think about it."},
                {"type": "thinking", "closed": True, "thinking": "2+2 = 4"},
                {
                    "type": "text",
                    "text": "The answer is 4.",
                },
            ],
        },
    ]

    conversation_with_thinking, _, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        content_format="openai",
    )

    expected_conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "text",
                    "text": "Only return the answer when you are confident.",
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is 2+2?"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think about it."},
                {"type": "text", "text": "2+2 = 4"},
                {"type": "text", "text": "The answer is 4."},
            ],
        },
    ]

    assert conversation_with_thinking == expected_conversation


@pytest.mark.parametrize("content_format", ["string", "openai"])
def test_file_content_part_is_replaced_with_text_note(
    mistral_model_config, monkeypatch, content_format
):
    """An unsupported `file` part becomes a text note instead of a 400.

    Chat clients replay their whole history each turn, so rejecting the part
    would wedge every later request in the same conversation. The note tells
    the model the attachment was not read.
    """
    logged: list[str] = []
    monkeypatch.setattr(
        chat_message_module.logger,
        "warning_once",
        lambda msg, *args, **kwargs: logged.append(msg % args),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize the attachment."},
                {
                    "type": "file",
                    "file": {
                        "filename": "spec.pdf",
                        "file_data": "data:application/pdf;base64,AAAA",
                    },
                },
            ],
        }
    ]

    conversation, _, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        content_format=content_format,
    )

    note = unsupported_attachment_note("spec.pdf")
    assert "spec.pdf" in note
    assert "无法被服务器解析" in note
    if content_format == "string":
        assert conversation == [
            {"role": "user", "content": f"Summarize the attachment.\n{note}"}
        ]
    else:
        assert conversation == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the attachment."},
                    {"type": "text", "text": note},
                ],
            }
        ]
    assert len(logged) == 1
    assert "'spec.pdf'" in logged[0]


def test_responses_input_file_part_is_replaced_with_text_note(mistral_model_config):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "spec.pdf",
                    "file_data": "data:application/pdf;base64,AAAA",
                },
            ],
        }
    ]

    conversation, _, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        content_format="string",
    )

    content = conversation[0]["content"]
    assert "spec.pdf" in content
    assert "无法被服务器解析" in content


def test_file_content_part_without_filename_uses_generic_note(mistral_model_config):
    messages = [
        {
            "role": "user",
            "content": [{"type": "file", "file": {"file_id": "file-abc"}}],
        }
    ]

    conversation, _, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        content_format="string",
    )

    note = unsupported_attachment_note()
    assert conversation == [{"role": "user", "content": note}]
    assert "spec.pdf" not in note


def test_unknown_content_part_type_is_still_rejected(mistral_model_config):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_u",
                    "image_url": {"url": "https://example.com/a.png"},
                },
            ],
        }
    ]

    with pytest.raises(VLLMValidationError, match="Unsupported chat content part type"):
        parse_chat_messages(messages, mistral_model_config, content_format="string")
