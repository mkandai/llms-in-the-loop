MAIN_PROMPT = """
Your task is to label entities in a given text written in {language}. Use the following labels for annotation:
* "O": Represents words that are not part of any named entity. 
* "B-PER": Indicates the beginning of a person's name.
* "I-PER": Used for tokens inside a person's name. 
* "B-ORG": Marks the beginning of an organization's name.
* "I-ORG": Tokens inside an organization's name.
* "B-LOC": Marks the beginning of a location (place) name.
* "I-LOC": Tokens inside a location name.
* "B-DATE": Marks the beginning of a date entity.
* "I-DATE": Tokens inside a date entity.
You will receive a list of tokens as the value for the 'input' key and text language as the value for the 'language' key in a JSON dictionary. Your task is to provide a list of named entity labels, where each label corresponds to a token. Output the tokens with their corresponding named entity labels in a JSON format, using the key 'output'. 'output' should contain a list of tokens and their entity labels in format (token, label).
{examples}
Note:
- The input tokens are provided in a list format and represent the text.
- Important: the output should be a list with the same length as the input list, where each element corresponds to the named entity label for the corresponding token. Do not change the order of tokens and do not skip them.
- The named entity labels are case-sensitive, so please provide them exactly as specified ("B-PER", "I-LOC", etc.). 
- Follow MUC-6 (Message Understanding Conference-6) Named Entity Recognition (NER) annotation guidelines.
Your task begins now! 
- Output JSON only. Enclose all tokens and tags in double brackets.
This is your sentence: {sentence}
"""

MAIN_PROMPT_FOR_BATCH = """
Your task is to label entities in a given text written in {language}. Use the following labels for annotation:
* "O": Represents words that are not part of any named entity. 
* "B-PER": Indicates the beginning of a person's name.
* "I-PER": Used for tokens inside a person's name. 
* "B-ORG": Marks the beginning of an organization's name.
* "I-ORG": Tokens inside an organization's name.
* "B-LOC": Marks the beginning of a location (place) name.
* "I-LOC": Tokens inside a location name.
* "B-DATE": Marks the beginning of a date entity.
* "I-DATE": Tokens inside a date entity.
You will receive a list of tokens as the value for the 'input' key and text language as the value for the 'language' key in a JSON dictionary. Your task is to provide a list of named entity labels, where each label corresponds to a token. Output the tokens with their corresponding named entity labels in a JSON format, using the key 'output'. 'output' should contain items that starts with 'record_' and each one of these items contain a list of tokens and their entity labels in format (token, label).
{examples}
Note:
- The input tokens are provided in a list format and represent the text.
- Important: the output should be a list with the same length as the input list, where each element corresponds to the named entity label for the corresponding token. Do not change the order of tokens and do not skip them.
- The named entity labels are case-sensitive, so please provide them exactly as specified ("B-PER", "I-LOC", etc.). 
- Follow MUC-6 (Message Understanding Conference-6) Named Entity Recognition (NER) annotation guidelines.
Your task begins now! 
- Output JSON only. Enclose all tokens and tags in double brackets.
This is your input: {inputs}
"""