from collections import Counter
from datasets import DatasetDict, concatenate_datasets
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger("datasets")

# Set the logging level to ERROR to suppress informational messages
logger.setLevel(logging.ERROR)


def filter_records_with_entities(dataset):
    """Filter records that contain at least one entity."""
    return dataset.filter(lambda record: any(label != 0 for label in record['ner_tags']))


def weighted_random_selection(dataset, N):
    """Select N records randomly with decreasing probability from the SORTED dataset."""
    np.random.seed(42)  # For reproducibility

    # Number of records in the dataset
    num_records = len(dataset)
    # Mu and sigma
    mu = num_records / 2; sigma = num_records / 4

    # Generate weights using a normal distribution function
    x = np.arange(num_records)
    weights = np.exp(-(x - mu)**2 / (2 * sigma**2))
    # Normalize weights to sum to 1
    weights /= weights.sum()

    # Select N indices based on the weights
    selected_indices = np.random.choice(num_records, size=N, replace=False, p=weights)
    # Select records from the dataset based on the chosen indices
    selected_records = dataset.select(selected_indices)

    return selected_records, weights


def add_non_entity_count(record):
    """Add a count of non-entity tokens (labels != 0) to each record."""
    entity_count = sum(1 for label in record['ner_tags'] if label != 0)
    record['entity_count'] = entity_count

    # Overall number of labels
    num_labels = sum(1 for label in record['ner_tags'] if label != -100)
    record['num_labels'] = num_labels

    # Proportion of entity_count to overall number of records
    record['% of entities'] = entity_count / num_labels

    return record


def get_labels(dataset):
    """Combine all entity labels into a single array"""
    labels = []

    # Loop through each split of the dataset ('train', 'validation', 'test')
    for i, split in enumerate(['train', 'validation', 'test']):
        # Extract the NER tags for the current split of the dataset
        tags_split = dataset[split]['ner_tags']

        # Iterate through each sentence (record) in the current split's NER tags
        for sentence in tags_split:
            # Add to the 'labels' list all tags from the sentence that are not equal to 0
            labels += [s for s in sentence if s != 0]
    return labels


def plot_ner_tag_distribution(labels, label_mapping, title=''):
    """Plots the distribution of NER tags."""

    # Count occurrences of each label
    ner_tags_counter = Counter([label_mapping[l] for l in labels])
    # Calculate normalized counts
    normalized_counts = {
        label: round(count / sum(ner_tags_counter.values()) * 100, 2)
        for label, count in ner_tags_counter.items()
    }

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    # Sort values by count in descending order
    labels, values = zip(*sorted(ner_tags_counter.items(), key=lambda x: x[1], reverse=True))
    axes.bar(labels, values)

    # Add numbers on top of each bar
    for label, value in zip(labels, values):
        axes.text(label, value, f'{value}\n({normalized_counts[label]}%)', ha='center', va='bottom')

    axes.set_xlabel('NER Tags')
    axes.set_ylabel('Counts')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def sample_for_model_selection(dataset, label_mapping, n_samples=100,
        min_percent_of_entities=0.05, max_percent_of_entities=0.5,
        train_key='train', concat_train_val_test=True, verbose=True):

    labels = get_labels(dataset)

    # Filter out records that do not contain any entities
    filtered_data = DatasetDict({
        split: filter_records_with_entities(dataset)
        for split, dataset in dataset.items()
    })

    # Add number of entities per record
    new_data = DatasetDict({
        split: dataset.map(add_non_entity_count)
        for split, dataset in filtered_data.items()
    })

    # Loads either the training set alone or combines the train, val, and test sets according to a Boolean switch.
    if concat_train_val_test:
        # Concatenate train, test and validation split
        merged_data = concatenate_datasets([dataset for dataset in new_data.values()])
    else:
        # Retrieve train data samples
        merged_data = new_data[train_key]

    # Select records where proportion of entities is in a specified range
    sorted_dataset = merged_data.filter(lambda x:
         min_percent_of_entities < x['% of entities'] < max_percent_of_entities) \
        .sort('entity_count', reverse=True)

    if verbose:
        # Plotting % of entities per sentence
        plt.figure(figsize=(6, 4))
        plt.hist(merged_data['% of entities'], bins=100, alpha=1)
        plt.title("% of Entities per sentence (excuded sentences without any entities)")
        plt.ylabel('Frequency')
        plt.axvline(x=min_percent_of_entities, color='black')
        plt.axvline(x=max_percent_of_entities, color='black')
        plt.grid(True)
        plt.show()

    top_n_records, sample_weights = weighted_random_selection(sorted_dataset, n_samples)
    if verbose:
        # Plotting the weights
        plt.figure(figsize=(6, 3))
        plt.plot(sample_weights, marker='o', linestyle='-', markersize=4, alpha=0.01)
        plt.title('Weights by record index using normal distribution')
        plt.xlabel('Record Index')
        plt.ylabel('Weight')
        plt.grid(True)        
        plt.show()

        # Plot distribution of entities in the initial dataset
        plot_ner_tag_distribution(labels, label_mapping,
                                  title='Distribution of entities in the initial dataset')

        # Plot distribution of entities in the sampled subset
        labels_top_n = [tag for sentence in top_n_records['ner_tags'] for tag in sentence if tag != 0]
        plot_ner_tag_distribution(labels_top_n, label_mapping,
                                  title='Distribution of entities in the sampled subset')

        # Plot number of entities in samples sentences
        plt.figure(figsize=(16, 6))
        # Plot 1: Number of Entities
        plt.subplot(1, 2, 1)
        plt.hist(top_n_records['entity_count'], bins=20, alpha=0.75)
        plt.title("Entity counts in the sampled subset")
        plt.xlabel('Entity Count')
        plt.ylabel('Frequency')
        plt.grid(True)
        # Plot 2: Percentage of Entities
        plt.subplot(1, 2, 2)
        plt.hist(top_n_records['% of entities'], bins=20, alpha=0.75, color='green')
        plt.title("% of entities in the sampled subset")
        plt.xlabel('% of Entities')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return top_n_records
