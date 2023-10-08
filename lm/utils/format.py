import json
import re
from random import random, shuffle
from enum import Enum
from typing import Literal, Optional

import pandas as pd
from itertools import zip_longest
from pydantic import BaseModel
# from pydantic import BaseModel, field_validator


SPECIAL_TOKENS = {
    'Patient': '<|patient|>',
    'Doctor': '<|doctor|>',
    'System': '<|system|>'
}


SYSTEM_PREFIX = '你是一个擅长心理咨询的人工智能助手，你将通过提问的方式一步一步引导用户回答，从而进行心理诊断。'


def compute_length(s: str) -> int:
    return len(re.findall(r"\w+", s)) // 5 + 1

class Role(str, Enum):
    prompter = 'Patient',
    assistant = 'Doctor'


class Utterance(BaseModel):
    text: str
    role: Role
    lang: str | None = None
    quality: float | None = None
    humor: float | None = None
    creativity: float | None = None
    context: str | None = None

    # @field_validator("quality", "humor", "creativity")
    # def between_0_1(cls, v, info) -> float:
    #     if v is not None and not (0 <= v <= 1):
    #         raise ValueError(f"Field {info.name} must be between 0 and 1. Received: {v}")
    #     return v

    def system_tag(
        self,
        eos_token: str,
        enabled: bool = True,
        property_dropout: float = 0.0,
        add_length: bool = True,
    ) -> str:
        if not enabled:
            return ""

        properties: list[tuple[float | str]] = []
        for k, v in dict(self).items():
            if v is not None and k in ["lang", "quality", "humor", "creativity"]:
                properties.append((k, v))

        if add_length:
            properties.append(("length", compute_length(self.text)))

        shuffle(properties)

        # ensure that potentially multi-line conext field comes last
        if self.context:
            properties.append(("context", self.context))

        fragments: list[str] = []
        for k, v in properties:
            if random() < property_dropout:
                continue

            if isinstance(v, float):
                fragments.append(f"{k}: {v:0.1f}")
            elif isinstance(v, str):
                if not v.isspace():  # ignore whitespace-only values
                    fragments.append(f"{k}: {v}")
            else:
                fragments.append(f"{k}: {v}")

        if len(fragments) == 0:
            return ""

        content = "\n".join(fragments)
        return f"{SPECIAL_TOKENS['System']}{content}\n{eos_token}"


class DatasetEntry(BaseModel):
    pass


class DatasetEntryLm(DatasetEntry):
    pass


class DatasetEntrySft(DatasetEntry):
    """Supervised fine-tuning conversation dataset entry"""

    conversation: list[Utterance]
    system_message: Optional[str] = None

    def get_formatted(
        self,
        eos_token: str,
        use_system_tag: bool = False,
        system_property_dropout: float = 0.5,
        system_add_length: bool = False,
    ) -> list[str]:
        output: list[str] = []

        for i, m in enumerate(self.conversation):
            if m.role == Role.prompter:
                if use_system_tag and i + 1 < len(self.conversation):
                    a = self.conversation[i + 1]
                    assert a.role == Role.assistant
                    system_tag = a.system_tag(
                        eos_token=eos_token,
                        property_dropout=system_property_dropout,
                        add_length=system_add_length,
                    )
                else:
                    system_tag = ""
                if i == 0 and self.system_message:
                    output.append(
                        f"{SPECIAL_TOKENS['System']}{self.system_message}{eos_token}{SPECIAL_TOKENS['Patient']}{m.text}{eos_token}{system_tag}"
                    )
                else:
                    output.append(f"{SPECIAL_TOKENS['Patient']}{m.text}{eos_token}{system_tag}")
            else:
                output.append(f"{SPECIAL_TOKENS['Doctor']}{m.text}{eos_token}")

        return output


def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    return data


def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.close()


def read_csv(data_path):
    return pd.read_csv(data_path)


def read_txt(data_path):
    f = open(data_path, 'r', encoding='utf-8')
    lines = f.readlines()
    return ''.join(lines)


def format_system_prefix(prefix, eos_token):
    return "{}{}{}".format(
        SPECIAL_TOKENS["System"],
        prefix,
        eos_token,
    )


def format_pairs(
    pairs: list[dict],
    eos_token: str,
    add_initial_reply_token: bool = False,
) -> list[str]:
    assert isinstance(pairs, list)
    conversations = [
        "{}{}{}".format(SPECIAL_TOKENS["Patient" if pairs[i]['speaker'] == 'patient' else "Doctor"], pairs[i]['content'], eos_token)
        for i in range(len(pairs))
    ]
    if add_initial_reply_token:
        conversations.append(SPECIAL_TOKENS["Doctor"])
    return conversations


def create_dataset_entry_qa(
    questions: list[str],
    answers: list[str] | list[list[str]],
    context: Optional[str] = None,
) -> DatasetEntry:
    """Helper function to create DatasetEntry objects (DatasetEntrySft or DatasetEntryRm) for simple
    Q&A datasets."""
    messages: list[Utterance] = []

    for q, a in zip_longest(questions, answers):
        messages.append(Utterance(text=q, role=Role.prompter))
        if isinstance(a, list):
            a = a[0]
        messages.append(Utterance(text=a, role=Role.assistant, context=context))

    return DatasetEntrySft(conversation=messages)
