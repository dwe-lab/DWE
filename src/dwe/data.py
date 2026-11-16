"""Small format adapters over Hugging Face Datasets; no experiment filtering."""

from datasets import load_dataset


def as_prompt(row, *, column, format, system_prompt=None, user_turn=-1):
    value = row[column]
    if format == "prompt":
        prompt = [{"role": "user", "content": value}] if isinstance(value, str) else value
    elif format == "sharegpt":
        questions = [turn["value"] for turn in value if turn["from"] in ("human", "user")]
        prompt = [{"role": "user", "content": questions[user_turn]}]
    else:
        raise ValueError(f"Unknown data format: {format}")
    if system_prompt:
        prompt = [{"role": "system", "content": system_prompt}, *prompt]
    return {"prompt": prompt}


def load_prompts(config):
    config = config.copy()
    adapter = config.pop("adapter")
    dataset = load_dataset(**config)
    return dataset.map(as_prompt, fn_kwargs=adapter, remove_columns=dataset.column_names)
