import json
from openai import APIStatusError, RateLimitError, APIConnectionError
import ast
from datasets import Dataset
from tqdm.notebook import tqdm
import regex as re


def ask_gpt(tokens, language, examples, openai_client, user_prompt, max_tokens=1000,
            temperature=0.7, model='gpt-4-1106-preview', system_prompt=None):
    """
    Generate named entity tags for a given sentence using the specified GPT model.

    Parameters:
    - tokens (str or list): list of tokens for which named entity recognition is desired.
    - language (str): The language of the input sentence.
    - openai_client (OpenAI API client object): The OpenAI client.
    - user_prompt (str, optional): Custom user prompt for the GPT model.
    - temperature (float, optional): A value between 0 and 1 that controls the randomness of the response.
      Lower values make the model more deterministic. Default is 0.3.
    - model (str, optional): The identifier of the GPT model to be used. Default is 'gpt-4-1106-preview'.
    - system_prompt (str, optional): Custom system prompt for the GPT model. If None, a default prompt is used.

    Returns:
    - ner_tags (list): A list of named entity tags generated for each token in the input sentence.
    - content (str): Text of the model response
    """
    # Convert token list to string
    sentence = str(tokens)

    if system_prompt is None:
        system_prompt = f"You are a named entity labelling expert in {language} language."

    # Format user prompt
    user_prompt = user_prompt.format(language=language, sentence=sentence, examples=examples)

    # Save query params
    query_params = {
        'model': model,
        'temperature': temperature,
        'messages': [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        'max_tokens': max_tokens,
    }

    if model == 'gpt-4-1106-preview' or model == 'gpt-4-0125-preview':  # Add additional params for new model
        query_params['response_format'] = {"type": "json_object"}

    try:
        # Query the model
        response = openai_client.chat.completions.create(**query_params)
    except APIConnectionError as e:
        raise Exception(f"The server could not be reached: {e.__cause__}")
    except RateLimitError as e:
        raise Exception(f"A 429 status code was received: {e}")
    except APIStatusError as e:
        raise Exception(f"Non-200-range status code received: {e.status_code}, {e.response}")

    try:
        # Extract NER tags from the response
        content = response.choices[0].message.content

        if model == 'gpt-4-1106-preview' or model == 'gpt-4-0125-preview':
            # Newer models provide json
            ner_tags = json.loads(content)
        else:
            # Extract json only
            match = re.search(r'\{(.*?)\}', content)

            if match:
                content = match.group(0)
                # Format output string to parse it as JSON
                ner_tags = json.loads(json.dumps(ast.literal_eval(content)))
            else:
                raise ValueError("No json found in model's response.")

    except Exception as e:
        print(response.choices[0].message.content)
        raise Exception(f"Cannot extract output from model's response: {e}")

    return ner_tags, content


def load_annotation_examples(json_filepath, language):
    """
    Load annotation examples in specified language.
    """
    with open(json_filepath, 'r') as json_file:
        examples = json.load(json_file)[language]

    return examples


def add_annotation_examples(json_filepath, language):
    """
    Create formatted string containing annotation examples
    in specified language for one sample.
    """
    examples = load_annotation_examples(json_filepath, language)

    example_str = f"""Example 1:
Input: {examples['example1']['input']}
Output: {{ 'output': {examples['example1']['output']} }}
Example 2:
Input: {examples['example2']['input']}
Output: {{ 'output': {examples['example2']['output']} }}"""

    return example_str


def add_annotation_examples_for_batch(json_filepath, language):
    """
    Create formatted string containing annotation examples
    in specified language for a batch of samples.
    """
    examples = load_annotation_examples(json_filepath, language)

    example_str = f"""Example:
Input: {examples['input']} 
Output: {{ 'output': {examples['output']} }}"""

    return example_str


def get_annotation(records, language, examples, openai_client, user_prompt, label_mapping,
                   temperature=0.3, model='gpt-4-1106-preview', system_prompt=None):
    """
    Annotate dataset subset using foundation model.
    """
    new_records = {'id': [], 'ner_tags': [], 'tokens': []}
    num_incorrect_predictions = 0

    for i in tqdm(range(len(records['id']))):  # For each record in the dataset
        # Extract tokens from the record
        tokens = records['tokens'][i]
        # Query foundation model
        try:
            ner, answer = ask_gpt(tokens, language, examples, openai_client, user_prompt,
                                  temperature=temperature, model=model, system_prompt=system_prompt)

            # Extract only tags
            tags = [n[1] for n in ner]
            # Convert NER tags to corresponding indexes
            ner_tags = [label_mapping[l] for l in tags]

            # Check if each token has a tag
            if len(ner_tags) != len(tokens):
                # print(f'Number of tokens and labels is not equal for this sentence: {tokens}')
                num_incorrect_predictions += 1
                continue
            else:
                new_records['id'].append(records['id'][i])
                new_records['tokens'].append(tokens)
                new_records['ner_tags'].append(ner_tags)
        except Exception as e:
            print(f'Something went wrong with this sentence: {tokens}')
            print(e)
            continue

    print(f'Number of incorrectly labeled records: {num_incorrect_predictions}')
    return Dataset.from_dict(new_records)
