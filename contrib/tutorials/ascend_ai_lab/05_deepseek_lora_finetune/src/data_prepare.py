"""多轮对话数据处理工具。"""
from datasets import load_dataset


def convert_conversation(conv):
    text = ""
    for turn in conv:
        if turn.get("system"):
            text += f"<|system|>\n{turn['system']}\n<|end|>\n"
        text += f"<|user|>\n{turn['input']}\n<|end|>\n"
        text += f"<|assistant|>\n{turn['output']}\n<|end|>\n"
    return text.strip()


def prepare_dataset(data_path, test_size=0.1):
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")
    dataset = dataset.map(
        lambda x: {"text": convert_conversation(x["conversation"])},
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=test_size)
    return split["train"], split["test"]
