from dwe.data import as_prompt


def test_upstream_prompt_excludes_solution():
    row = {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "reference answer"},
        ]
    }
    assert as_prompt(row, column="conversations", format="sharegpt") == {
        "prompt": [{"role": "user", "content": "question"}]
    }


def test_prompt_format_preserves_conversation():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert as_prompt({"p": messages}, column="p", format="prompt")["prompt"] == messages
