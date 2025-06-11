import random

from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer

from .request import APIRequest

# dataset_name: ('hg dataset path', 'subset', 'split', 'prompt', 'output')
dataset_cols = {
    "alpaca": ("tatsu-lab/alpaca", None, "train", "instruction", "output"),
    "humaneval":
    ("openai/openai_humaneval", None, "test", "prompt", "canonical_solution"),
}


def load_and_preprocess_dataset(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int,
):
    if dataset_name in dataset_cols.keys():
        (dataset_path, subset, split, prompt_col,
         output_col) = dataset_cols[dataset_name]
    else:
        raise RuntimeError(
            f"The dataset '{dataset_name}' does not specify which column to"
            "tokenize. If you want to use this dataset, please define"
            "'dataset_cols' in 'request_generator.py'")

    dataset = load_dataset(dataset_path, subset, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(data):
        tokenized_input = tokenizer(data[prompt_col],
                                    truncation=True,
                                    max_length=max_length)
        tokenized_output = tokenizer(data[output_col],
                                     truncation=True,
                                     max_length=max_length)

        input_ids = tokenized_input["input_ids"]
        output_ids = tokenized_output["input_ids"]

        return {
            "prompt": data[prompt_col],
            "output": data[output_col],
            "input_ids": input_ids,
            "output_ids": output_ids,
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def generate_requests(
    dataset_name: str,
    tokenizer_name: str,
    max_seq_len: int,
    num_requests: int,
    num_samples: int,
    ignore_eos: bool,
) -> List[APIRequest]:

    dataset = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_seq_len,
    )

    requests: List[APIRequest] = []
    num_requests_counter = 0
    while num_requests_counter < num_requests:
        dataset = dataset.shuffle()
        for data in dataset.to_iterable_dataset():
            input_len = len(data['input_ids'])
            output_len = len(data['output_ids'])
            if (input_len + output_len) > max_seq_len:
                continue

            request = APIRequest(
                prompt=data['prompt'],
                num_samples=num_samples,
                max_output_len=output_len,
                ignore_eos=ignore_eos,
            )
            requests.append(request)

            num_requests_counter += 1

            if num_requests_counter >= num_requests:
                break

    return requests


def generate_radom_requests(
    tokenizer_name: str,
    max_input_len: int,
    max_output_len: int,
    max_seq_len: int,
    num_requests: int,
    num_samples: int,
) -> List[APIRequest]:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    requests: List[APIRequest] = []
    while len(requests) < num_requests:
        input_len = random.randint(min(max_input_len, 4), max_input_len)
        output_len = random.randint(min(max_output_len, 32), max_output_len)
        if (input_len + output_len) > max_seq_len:
            continue

        input_ids = [3] * input_len
        prompt = tokenizer.decode(input_ids)

        request = APIRequest(
            prompt=prompt,
            num_samples=num_samples,
            max_output_len=output_len,
            ignore_eos=True,
        )
        requests.append(request)

    return requests
