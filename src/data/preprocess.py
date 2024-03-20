import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from datasets import DatasetDict
from src.data.sample import sample_for_model_selection


def align_labels(tokens, labels, tokenizer):
    """
    Aligns token labels with a tokenizer's output, considering subword tokenization.

    Args:
        tokens (List[str]): A list of word tokens that are not yet tokenized by the model's tokenizer.
        labels (List[int]): A list of integer labels corresponding to each token in `tokens`.
        tokenizer: The tokenizer that will be used to tokenize the `tokens`. This should be
                   a pre-initialized tokenizer from a library like Hugging Face's transformers.

    Returns:
        Dict[str, List]: A dictionary with keys corresponding to tokenizer output (input_ids, attention_mask, etc.)
                         and a key 'labels' containing the aligned labels ready for model training.

    Raises:
        AssertionError: If the length of `tokens` and `labels` do not match.
    """
    # Ensure the tokens and labels lists are of the same length
    assert len(tokens) == len(labels), 'The length of tokens and labels must be the same.'

    # Create a mapping from token indices to their corresponding labels
    idx_to_label = {i: label for i, label in enumerate(labels)}

    # Special token ID that indicates a label to be ignored during loss computation.
    ignored_label_id = -100

    # Assign the ignore label ID to None to handle alignment of word-piece tokens to original tokens
    idx_to_label[None] = ignored_label_id

    # Tokenize the input tokens. `is_split_into_words=True` indicates that the input is pre-tokenized.
    tokenized_input = tokenizer(tokens, is_split_into_words=True)

    # Align the labels with the tokens using the 'word_ids' provided by the tokenizer
    tokenized_input['labels'] = [
        idx_to_label.get(i, ignored_label_id) for i in tokenized_input.word_ids()
    ]

    return tokenized_input


def align_labels_for_many_records(records, tokenizer):
    """
    Process multiple records by aligning named entity recognition (NER) tags with tokenized text.

    Args:
        records (dict): A dictionary containing information about multiple records.
                        It should have 'id', 'tokens', and 'ner_tags' fields for each record.
        tokenizer: An instance of a tokenizer used for tokenizing the input text.

    Returns:
        dict: The input records object with additional fields for labels, attention masks, and input IDs.
    """
    # Initialize lists to store processed information for each record
    labels, attention_mask, input_ids = [], [], []

    # Loop over each record
    for i in range(len(records['id'])):
        # Call the align_labels function to align NER tags with tokenized text
        result = align_labels(records['tokens'][i], records['ner_tags'][i], tokenizer)

        # Append the results to the respective lists
        labels.append(list(result['labels']))
        attention_mask.append(result['attention_mask'])
        input_ids.append(result['input_ids'])

    # Update the records object with the processed information
    records['labels'] = labels
    records['attention_mask'] = attention_mask
    records['input_ids'] = input_ids

    # Return the updated records object
    return records


class TorchDataset(Dataset):
    def __init__(self, hf_dataset, max_length=None, padding_value=0):
        self.hf_dataset = hf_dataset
        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # Cropping or padding the items individually
        labels = torch.tensor(item['labels'])
        attention_mask = torch.tensor(item['attention_mask'])
        input_ids = torch.tensor(item['input_ids'])

        if self.max_length is not None:
            # Crop if the sequence is too long
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            input_ids = input_ids[:self.max_length]

            # Padding if the sequence is too short
            labels_padded = torch.full((self.max_length,), self.padding_value, dtype=labels.dtype)
            attention_mask_padded = torch.full((self.max_length,), self.padding_value, dtype=attention_mask.dtype)
            input_ids_padded = torch.full((self.max_length,), self.padding_value, dtype=input_ids.dtype)

            labels_padded[:len(labels)] = labels
            attention_mask_padded[:len(attention_mask)] = attention_mask
            input_ids_padded[:len(input_ids)] = input_ids

            labels = labels_padded
            attention_mask = attention_mask_padded
            input_ids = input_ids_padded

        return {'labels': labels, 'attention_mask': attention_mask, 'input_ids': input_ids}


def split_dataset(dataset, split_ratio=0.7, shuffle=True):
    """Split PyTorch dataset."""
    # Determine the size of each split
    dataset_size = len(dataset)
    split = int(np.floor(split_ratio * dataset_size))

    # Generate indices and shuffle them if needed
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    # Split indices into two parts
    train_indices, val_indices = indices[:split], indices[split:]

    # Create two subsets for the splits
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def split_for_active_learning(dataset, train_key='train', split_ratio=0.7, verbose=False):
    """
    Splits the training dataset into two parts for initial model training and active learning.

    Args:
    dataset (DatasetDict): The original dataset dictionary.
    train_key (str): The key name for the training dataset, defaults to 'train'.
    split_ratio (float): The ratio of the split for the initial training dataset, defaults to 0.7.

    Returns:
    DatasetDict: A new dataset dictionary with the training data split into two parts.
    """
    # Ensure split ratio is valid
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1")

    # Shuffle the training data
    train_dataset = dataset[train_key].shuffle(seed=42)

    # Determine the split index
    split_index = int(len(train_dataset) * split_ratio)

    # Split the dataset
    initial_training_data = train_dataset.select(range(split_index))
    active_learning_data = train_dataset.select(range(split_index, len(train_dataset)))

    if verbose:
        print(f"Split the dataset into initial training (first {split_ratio * 100}%) and active learning (remaining {100 - split_ratio * 100}%).")
        print(f"Overall training dataset size: {len(train_dataset)}")
        print(f"Initial training dataset size: {len(initial_training_data)}")
        print(f"Active learning dataset size: {len(active_learning_data)}")

    # Create new dataset dict
    new_dataset_dict = DatasetDict({
        'initial_training': initial_training_data,
        'active_learning': active_learning_data
    })

    # Add other datasets (like validation and test) if they exist
    for key in dataset.keys():
        if key != train_key:
            new_dataset_dict[key] = dataset[key]

    return new_dataset_dict


def balanced_split_for_active_learning(dataset, label_mapping, train_key='train',
                                       split_ratio=0.7, verbose=False):
    """
    Splits the training dataset into two parts for initial model training and active learning.
    Makes initial training set more balanced for initial model training. 

    Args:
    dataset (DatasetDict): The original dataset dictionary.
    train_key (str): The key name for the training dataset, defaults to 'train'.
    split_ratio (float): The ratio of the split for the initial training dataset, defaults to 0.7.

    Returns:
    DatasetDict: A new dataset dictionary with the training data split into two parts.
    """
    # Ensure split ratio is valid
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1")

    # Shuffle the training data
    train_dataset = dataset[train_key].shuffle(seed=42)

    # Determine the split index
    split_index = int(len(train_dataset) * split_ratio)

    # Create initial training sample set
    try:
        initial_training_data = sample_for_model_selection(dataset=dataset,
                                label_mapping=label_mapping,
                                n_samples=split_index,
                                train_key=train_key,
                                concat_train_val_test=False,
                                verbose=verbose)
    except:
        raise ValueError("initial_train_size is too large.")

    # Extract IDs from both original train dataset and initial training sample set
    train_dataset_ids = set(train_dataset['id'])
    initial_training_data_ids = set(initial_training_data['id'])

    # Exclude IDs from train dataset that are present in initial training set
    remaining_ids = train_dataset_ids - initial_training_data_ids

    # Create active learning sample set by filtering initial training samples from original train dataset
    active_learning_data = train_dataset.filter(lambda sample: sample['id'] in remaining_ids)

    if verbose:
        print(f"Split the dataset into initial training (first {split_ratio * 100}%) and active learning (remaining {100 - split_ratio * 100}%).")
        print(f"Overall training dataset size: {len(train_dataset)}")
        print(f"Initial training dataset size: {len(initial_training_data)}")
        print(f"Active learning dataset size: {len(active_learning_data)}")

    # Create new dataset dict
    new_dataset_dict = DatasetDict({
        'initial_training': initial_training_data,
        'active_learning': active_learning_data
    })

    # Add other datasets (like validation and test) if they exist
    for key in dataset.keys():
        if key != train_key:
            new_dataset_dict[key] = dataset[key]

    return new_dataset_dict
