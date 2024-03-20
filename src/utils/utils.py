import torch
import numpy as np
from scipy.stats import entropy
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import requests
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score


def logits_to_probabilities(logits):
    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(logits)
    return probabilities


def probabilities_to_labels(probabilities):
    # tensor has the shape [batch_size, sequence_length, num_classes]
    # apply argmax to last dimension
    class_labels = torch.argmax(probabilities, dim=-1)
    return class_labels


def predict_sequence_max_uncertainty(model, dataloder_unlabeled, device, fraction=0.1):
    """
    Predict the class probabilities for each item in the sequences and calculate
    the maximum entropy across the sequence for each instance in the batch.

    :param model: Trained machine learning model.
    :param unlabeled_data: Data on which predictions need to be made (DataLoader).
    :param fraction: Fraction of data to return based on uncertainty (e.g., 0.10 for 10%).
    :return: Indices of the most uncertain instances.
    """
    model.eval()  # Set the model to evaluation mode

    max_entropies = []
    batch_indices = []

    with torch.no_grad():  # Disable gradients
        for batch_idx, batch in enumerate(dataloder_unlabeled):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            probabilities = logits_to_probabilities(logits)

            # Calculate entropy for each item's probability distribution and find the maximum across the sentence
            max_entropy_per_instance = np.array([entropy(prob.T).mean() for prob in probabilities.cpu().detach().numpy()])
            max_entropies.extend(max_entropy_per_instance)
            batch_indices.extend([(batch_idx, i) for i in range(len(max_entropy_per_instance))])

    max_entropies = np.array(max_entropies)
    num_samples = int(len(max_entropies) * fraction)

    print(f'Selecting {num_samples} with highest uncertainty out of {len(max_entropies)}')

    # Find the indices of the 'num_samples' instances with the highest maximum entropy
    most_uncertain_indices = np.argsort(-max_entropies)[:num_samples]
    selected_indices = [batch_indices[idx] for idx in most_uncertain_indices]

    return selected_indices


def batch_indices_to_global_indices(batch_indices, batch_size):
    """
    Convert batch-specific indices to global indices in the dataset.
    """
    global_indices = [batch_idx * batch_size + item_idx for batch_idx, item_idx in batch_indices]
    return global_indices


def print_classification_report(config, model, test_loader, device, ignore_index=-100, ignore_class=None):
    model.eval()  # Set the model to evaluation mode

    true_labels = []
    predictions = []

    with torch.no_grad():  # No need to track gradients
        for batch in tqdm(test_loader, desc=f"[test]", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)

            predicted = torch.argmax(logits, dim=-1)

            # Extend the true labels and predictions lists
            true_labels.append(batch['labels'].detach())
            predictions.append(predicted.detach())

    # Concatenate all batches
    all_predictions = np.concatenate([batch.cpu().numpy() for batch in predictions])
    all_labels = np.concatenate([batch.cpu().numpy() for batch in true_labels])

    if ignore_class is not None:
        # Flatten and filter out ignore_index and ignore_class
        mask = (all_labels != ignore_index) & (all_labels != ignore_class)
        true_labels = all_labels[mask].flatten()
        pred_labels = all_predictions[mask].flatten()
    else:
        # Flatten and filter out ignore_index
        true_labels = all_labels[all_labels != ignore_index].flatten()
        pred_labels = all_predictions[all_labels != ignore_index].flatten()

    # Convert label indexes to actual indexes
    true_labels = [config['label_mapping'][l] for l in true_labels]
    pred_labels = [config['label_mapping'][l] for l in pred_labels]
    
    # Calculate F1 score
    report = classification_report(true_labels, pred_labels)
    print(report)


def calculate_micro_f1_for_batches(all_predictions, all_labels, ignore_index=-100, ignore_class=None):
    """
    Calculate the Micro-averaged F1 Score for batched data.

    Args:
    all_predictions (list of torch.Tensor): List of predicted labels for each batch.
    all_labels (list of torch.Tensor): List of true labels for each batch.
    ignore_index (int): Label index to ignore when calculating F1 (e.g., for padding tokens).
    ignore_class (int): Class index to ignore when calculating F1.

    Returns:
    float: The Micro-averaged F1 Score.
    """
    # Concatenate all batches
    all_predictions = np.concatenate([batch.cpu().numpy() for batch in all_predictions])
    all_labels = np.concatenate([batch.cpu().numpy() for batch in all_labels])

    if ignore_class is not None:
        # Flatten and filter out ignore_index and ignore_class
        mask = (all_labels != ignore_index) & (all_labels != ignore_class)
        true_labels = all_labels[mask].flatten()
        pred_labels = all_predictions[mask].flatten()
    else:
        # Flatten and filter out ignore_index
        true_labels = all_labels[all_labels != ignore_index].flatten()
        pred_labels = all_predictions[all_labels != ignore_index].flatten()

    # Calculate Micro F1 Score
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    return micro_f1


def calculate_macro_f1_for_batches(all_predictions, all_labels, ignore_index=-100, ignore_class=None):
    """
    Calculate the Macro-averaged F1 Score for batched data.

    Args:
    all_predictions (list of torch.Tensor): List of predicted labels for each batch.
    all_labels (list of torch.Tensor): List of true labels for each batch.
    ignore_index (int): Label index to ignore when calculating F1 (e.g., for padding tokens).
    ignore_class (int): Class index to ignore when calculating F1.

    Returns:
    float: The Macro-averaged F1 Score.
    """
    # Concatenate all batches
    all_predictions = np.concatenate([batch.cpu().numpy() for batch in all_predictions])
    all_labels = np.concatenate([batch.cpu().numpy() for batch in all_labels])

    if ignore_class is not None:
        # Flatten and filter out ignore_index and ignore_class
        mask = (all_labels != ignore_index) & (all_labels != ignore_class)
        true_labels = all_labels[mask].flatten()
        pred_labels = all_predictions[mask].flatten()
    else:
        # Flatten and filter out ignore_index
        true_labels = all_labels[all_labels != ignore_index].flatten()
        pred_labels = all_predictions[all_labels != ignore_index].flatten()

    # Calculate Macro F1 Score
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    return macro_f1


def calculate_consistency_score(predictions, ground_truth):
    """
    Calculate the Consistency Score against ground truth.
    predictions: list of lists, where each sublist represents one model inference's labels
    ground_truth: list of labels representing the ground truth for each token
    """
    total_tokens = len(ground_truth)
    consistency_score = 0
    num_predictions = 0

    for inference in predictions:
        # Extract predicted labels
        inference = [t[1] if len(t) > 1 and t[1] is not None else None for t in inference]
        # If length is different (tokens skipped) - consider complete mismatch
        if len(inference) != len(ground_truth):
            num_predictions += 1
            continue
        # Count number of matching tokens
        correct_count = sum(pred == gt for pred, gt in zip(inference, ground_truth))
        consistency_score += correct_count
        num_predictions += 1

    # Divide count number of matching tokens by total number of tokens
    consistency_score = consistency_score / (total_tokens * num_predictions)
    return round(consistency_score, 4) * 100


def get_api_usage(api_key, date, model_name=None):
    """
    Fetches and prints the API usage information for a given API key and date,
    summing up costs for each model name and providing overall usage. If a model_name
    is provided, returns the total cost for that model only.

    Parameters:
    api_key (str): The OPENAI API key.
    date (datetime.date): The date for which to get usage data.
    model_name (str, optional): The name of the model for which to calculate the total cost.

    Returns:
    float: Total cost for the specified model if model_name is provided, otherwise None.
    """
    # API headers
    headers = {'Authorization': f'Bearer {api_key}'}

    # API endpoint
    url = 'https://api.openai.com/v1/usage'

    # Parameters for API request
    params = {'date': date.strftime('%Y-%m-%d')}

    # Send API request and get response
    response = requests.get(url, headers=headers, params=params)
    usage_data = response.json()['data']

    # Initialize totals for each model and overall
    totals = {}  # Key is model name
    overall_totals = {
        'n_generated_tokens': 0,
        'n_context_tokens': 0,
        'total_cost': 0
    }

    # Process usage data
    for data in usage_data:
        model = data['snapshot_id']  # Adjusted to use the correct key for model name
        n_generated_tokens = data['n_generated_tokens_total']
        n_context_tokens = data['n_context_tokens_total']

        # Ensure model is in totals dictionary
        if model not in totals:
            totals[model] = {
                'n_generated_tokens': 0,
                'n_context_tokens': 0
            }

        # Sum tokens by model
        totals[model]['n_generated_tokens'] += n_generated_tokens
        totals[model]['n_context_tokens'] += n_context_tokens

        # Sum up overall totals without considering model name
        overall_totals['n_generated_tokens'] += n_generated_tokens
        overall_totals['n_context_tokens'] += n_context_tokens

    # Define cost per token for known models (price from api)
    costs_per_token = {
        'gpt-4-0125-preview': {'input': 0.01 / 1000, 'output': 0.03 / 1000},
        'gpt-4': {'input': 0.03 / 1000, 'output': 0.06 / 1000},
        'gpt-4-0613': {'input': 0.03 / 1000, 'output': 0.06 / 1000},
        'gpt-4-1106-vision-preview': {'input': 0.03 / 1000, 'output': 0.06 / 1000},
        'gpt-3.5-turbo-0125': {'input': 0.0005 / 1000, 'output': 0.0015}
    }

    # Calculate and print costs by model and overall
    if model_name is not None:
        input_cost_per_token = costs_per_token.get(model_name, {'input': 0, 'output': 0})['input']
        output_cost_per_token = costs_per_token.get(model_name, {'input': 0, 'output': 0})['output']

        try:
          total_input_tokens = totals[model_name]['n_context_tokens']
          total_output_tokens = totals[model_name]['n_generated_tokens']
        except:
          total_input_tokens = 0
          total_output_tokens = 0

          total_cost_input = total_input_tokens * input_cost_per_token
          total_cost_output = total_output_tokens * output_cost_per_token

          return total_cost_input + total_cost_output

    else:
        # Print costs for all models
        for model, counts in totals.items():
            input_cost_per_token = costs_per_token.get(model, {'input': 0, 'output': 0})['input']
            output_cost_per_token = costs_per_token.get(model, {'input': 0, 'output': 0})['output']

            total_cost_input = counts['n_context_tokens'] * input_cost_per_token
            total_cost_output = counts['n_generated_tokens'] * output_cost_per_token
            model_total_cost = total_cost_input + total_cost_output

            # Update overall total cost
            overall_totals['total_cost'] += model_total_cost

            # Print out stats per model
            print(f"Model: {model.upper()}\n" + '-' * 35)
            print("{:<20} {:>15,}".format("Total Input Tokens", counts['n_context_tokens']))
            print("{:<20} {:>15,}".format("Total Output Tokens", counts['n_generated_tokens']))
            print("{:<20} {:>15.2f}".format("Input Cost ($)", total_cost_input))
            print("{:<20} {:>15.2f}".format("Output Cost ($)", total_cost_output))
            print("{:<20} {:>15.2f}".format("Total Cost ($)", model_total_cost))
            print("\n")

        # Print overall usage
        print("Overall Usage:")
        print(f"Total number of tokens as input: {overall_totals['n_context_tokens']:,}")
        print(f"Total number of tokens as output: {overall_totals['n_generated_tokens']:,}")
        print(f"Total cost across all models on {date}: ${overall_totals['total_cost']:.2f}\n")


def print_seqeval_classification_report(config, model, test_loader, device, ignore_index=-100):
    model.eval()  # Set the model to evaluation mode

    true_labels = []
    predictions = []

    with torch.no_grad():  # No need to track gradients
        for batch in tqdm(test_loader, desc=f"[test]", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)

            predicted = torch.argmax(logits, dim=-1)

            # Extend the true labels and predictions lists
            true_labels += batch['labels'].detach().cpu().tolist()
            predictions += predicted.detach().cpu().tolist()

    for i in range(len(true_labels)):  # For each record
        # Remove padding index
        true_labels[i] = [label for label in true_labels[i] if label != ignore_index]
        predictions[i] = [pred for label, pred in zip(true_labels[i], predictions[i]) if label != ignore_index]

        # Convert label indexes to actual indexes
        true_labels[i] = [config['label_mapping'][l] for l in true_labels[i]]
        predictions[i] = [config['label_mapping'][l] for l in predictions[i]]

    # Calculate F1 score
    print(seqeval_classification_report(true_labels, predictions))


def calculate_seqeval_f1_for_batches(config, all_predictions, all_labels, ignore_index=-100):
    true_labels = []
    predictions = []

    for batch in all_predictions:
        predictions += batch.cpu().tolist()
    for batch in all_labels:
        true_labels += batch.cpu().tolist()

    for i in range(len(true_labels)):  # For each record
        # Remove padding index
        true_labels[i] = [label for label in true_labels[i] if label != ignore_index]
        predictions[i] = [pred for label, pred in zip(true_labels[i], predictions[i]) if label != ignore_index]

        # Convert label indexes to actual indexes
        true_labels[i] = [config['label_mapping'][l] for l in true_labels[i]]
        predictions[i] = [config['label_mapping'][l] for l in predictions[i]]

    # Calculate Macro F1 Score
    micro_f1 = seqeval_f1_score(true_labels, predictions)
    return micro_f1
