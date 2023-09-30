import json
import pandas as pd


SPECIAL_TOKENS = {
    'Patient': '<|patient|>',
    'Doctor': '<|doctor|>',
    'System': '<|system|>'
}


def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    return data


def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, save_path, ensure_ascii=False, indent=1)
        f.close()


def read_csv(data_path):
    return pd.read_csv(data_path)


def format_system_prefix(prefix, eos_token):
    return "{}{}{}".format(
        SPECIAL_TOKENS["System"],
        prefix,
        eos_token,
    )


def format_pairs(
    pairs: list[str],
    eos_token: str,
    add_initial_reply_token: bool = False,
) -> list[str]:
    assert isinstance(pairs, list)
    conversations = [
        "{}{}{}".format(SPECIAL_TOKENS["Patient" if i % 2 == 0 else "Doctor"], pairs[i], eos_token)
        for i in range(len(pairs))
    ]
    if add_initial_reply_token:
        conversations.append(SPECIAL_TOKENS["Doctor"])
    return conversations
