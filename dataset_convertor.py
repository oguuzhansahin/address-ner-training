import json
from spacy.training import biluo_to_iob
from config import label2id
from typing import List, Dict
from datasets import Dataset


def read_json(json_name: str
              ) -> dict:
    with open(json_name, "r") as f:
        data = json.load(f)
    return data


def fix_labels_error(json_data: Dict
                     ) -> [List, List]:
    cleaned_tokens, cleaned_labels = [], []
    for annotation in json_data:
        tokens, labels = zip(*[(pair["token"], pair["label"]) for pair in annotation["items"]])
        tokens, labels = list(tokens), list(labels)

        # Some labels are "-", due to partial word annotation.
        broken_labels = labels.count("-")
        if broken_labels != 0:
            num_of_token = len(tokens)
            if broken_labels / num_of_token < 0.4:
                broken_indices = [i for i, x in enumerate(labels) if x == "-"]

                labels = [label for idx, label in enumerate(labels) if idx not in broken_indices]
                tokens = [token for idx, token in enumerate(tokens) if idx not in broken_indices]
            else:
                continue

        labels = biluo_to_iob(labels)
        #labels = [label2id(label) for label in labels]

        assert len(labels) == len(tokens)

        cleaned_tokens.append(tokens)
        cleaned_labels.append(labels)

    return cleaned_tokens, cleaned_labels


def convert_json_to_hug_dataset(json_name: str,
                                save_dataset:bool,
                                save_name: str,
                                test_size: float = 0.2
                               ) -> Dataset:
    json_data = read_json(json_name)


    cleaned_tokens, cleaned_labels = fix_labels_error(json_data)

    dataset_dict = {"id": [i for i in range(len(cleaned_tokens))],
                    "tokens": cleaned_tokens,
                    "ner_tags": cleaned_labels
                    }

    dataset = Dataset.from_dict(dataset_dict)
    if test_size:
        dataset = dataset.train_test_split(test_size = 0.2, seed = 42)
    if save_dataset:
        dataset.save_to_disk(save_name)

    return dataset