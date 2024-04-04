import re
import json

HF_ACCESS_TOKEN = ""  # Add your huggingface access token here.

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f.readlines() if line.strip()]

def dump_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d)+'\n')

def replace_numbers_with_arguments(input_text):
    num_to_arg_mapping = {}
    pattern = r'\d+(?:[\.\,]\d+)?'
    def replace(match):
        num = match.group()
        arg = f"arg{chr(ord('A')+len(num_to_arg_mapping))}"
        num_to_arg_mapping[arg] = num
        return arg
    modified_text = re.sub(pattern, replace, input_text)
    return modified_text, num_to_arg_mapping

def replace_arguments_with_numbers(modified_text, num_to_arg_mapping):
    original_text = modified_text
    for arg, num in num_to_arg_mapping.items():
        original_text = original_text.replace(arg, num)
    return original_text