import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
from tqdm import tqdm

import textstat
from transformers import AutoTokenizer
from sentence_splitter import SentenceSplitter

from utils import (
    replace_numbers_with_arguments,
    HF_ACCESS_TOKEN,
)

######################################################################

WORD_ARGS_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

WORD_ARGS_TENS = {
    "twenty": 20, "thirty": 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}

WORD_ARGS_TEENS = {
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
}

WORD_ARGS_21_99 = {f"{tens} {ones}".strip(): t + o for tens, t in WORD_ARGS_TENS.items() for ones, o in WORD_ARGS_ONES.items()}

WORD_ARGS_FRAC = {
    "half": 1/2, "one third": 1/3, "a third": 1/3, "two third": 2/3, "one fourth": 1/4, "a fourth": 1/4, "three fourth": 3/4,
}

WORD_NUMERALS = {"zero": 0}

WORD_NUMERALS.update(WORD_ARGS_ONES)
WORD_NUMERALS.update({"ten": 10})
WORD_NUMERALS.update(WORD_ARGS_TEENS)
WORD_NUMERALS.update(WORD_ARGS_TENS)
WORD_NUMERALS.update(WORD_ARGS_21_99)
WORD_NUMERALS.update(WORD_ARGS_FRAC)

######################################################################

# Features catering to questions (and gold solutions)

def token_length(arr):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token=HF_ACCESS_TOKEN if HF_ACCESS_TOKEN!="" else None)
    return [len(tokenizer.tokenize(a)) for a in tqdm(arr,desc="token_length()")]

def sentence_length(arr):
    s_splitter = SentenceSplitter(language='en')
    return [len(s_splitter.split(text=a)) for a in tqdm(arr,desc="sentence_length()")]

def word_length(arr):
    return [len(a.split(" ")) for a in tqdm(arr,desc="word_length()")]

def get_question_args(q):
    numerical_args = len(re.findall(r'\d+(?:[\.\,]\d+)?',q))
    q = re.sub(r'[ ]+',' ',re.sub(r'[^a-z]',' ',q.strip().lower()))
    word_args = sum([len(re.findall(word,q)) for word,num in WORD_NUMERALS.items()])
    return numerical_args, word_args

def arg_count(arr):
    num_arg_count, word_arg_count =  list(zip(*[get_question_args(a) for a in tqdm(arr,desc="arg_count()")]))
    return [n+w for n,w in zip(num_arg_count, word_arg_count)], num_arg_count, word_arg_count

def flesch_reading_ease(arr):
    return [textstat.flesch_reading_ease(a) for a in tqdm(arr,desc="flesch_reading_ease()")]

def flesch_kinkaid_grade(arr):
    return [textstat.flesch_kincaid_grade(a) for a in tqdm(arr,desc="flesch_kinkaid_grade()")]

def mean_word_rank(arr, omit_numerical_tokens=False):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token=HF_ACCESS_TOKEN if HF_ACCESS_TOKEN!="" else None)
    return [np.mean(tokenizer.convert_tokens_to_ids([t for t in tokenizer.tokenize(a) if not omit_numerical_tokens or not re.match(r'[0-9]+',t)])) for a in tqdm(arr,desc="mean_word_rank()")]

#---------------------------------------------------------------------

# Features requiring 'parse_and_ner_data' object

def constituency_tree_depth(parse_and_ner_data):
    return [max([s.constituency.depth() for s in a.sentences]) for a in tqdm(parse_and_ner_data,desc="constituency_tree_depth()")]

def np_count(parse_and_ner_data):
    return [sum([len((re.findall(r"\(NP",str(s.constituency)))) for s in a.sentences]) for a in tqdm(parse_and_ner_data,desc="np_count()")]

def prp_count(parse_and_ner_data):
    return [sum([t.to_dict()[0]["xpos"]=="PRP" if "xpos" in t.to_dict()[0] else False for s in a.sentences for t in s.tokens]) for a in tqdm(parse_and_ner_data,desc="prp_count()")]

def unique_np_count(parse_and_ner_data):
    _unique_np_count = []
    for a in tqdm(parse_and_ner_data,desc="unique_np_count()"):
        NPs = []
        for s in a.sentences:
            for t in s.tokens:
                t = t.to_dict()[0]
                if "deprel" in t and t["deprel"] == "nsubj":
                    NPs.append(t["text"])
        feat = len(set(NPs))
        _unique_np_count.append(feat)
    return _unique_np_count

def multi_np_count(parse_and_ner_data):
    _multi_np_count = []
    for q in tqdm(parse_and_ner_data,desc="multi_np_count()"):
        NPs = []
        NNPs = []
        for s in q.sentences:
            for t in s.tokens:
                t = t.to_dict()[0]
                if "xpos" in t and t["xpos"] == "PRP":
                    NPs.append(t["text"].lower())
                if "xpos" in t and t["xpos"] == "NNP":
                    NNPs.append(t["text"].lower())
        # print(set(NNPs), set(NPs))
        feat = max(len(set(NNPs)) - 1, 0) * len(set(NPs))
        _multi_np_count.append(feat)
    return _multi_np_count

#=====================================================================

# Features catering to gold solutions 

def get_all_math_operations(gs):
    lines = re.findall(r'<<([^>>]+)>>',gs)
    ops = [op for l in lines for op in re.sub(r'[^\+\-\*\/\(\)]','',l)]
    return ops

def op_count(arr,operation):
    return [len([op for op in get_all_math_operations(a) if op==operation]) for a in tqdm(arr,desc=f"op_count()['{operation}']")]

def unique_op_count(arr):
    return [len(set(list(get_all_math_operations(a)))) for a in tqdm(arr,desc="unique_op_count()")]

def op_diversity(arr):
    return [(1+len(set(list(ops))))/(1+len(ops)) for a in tqdm(arr,desc="op_diversity()") if (ops := get_all_math_operations(a))!=None]

def mean_numerical_word_rank(arr):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token=HF_ACCESS_TOKEN if HF_ACCESS_TOKEN!="" else None)
    return [np.mean([tokenizer.convert_tokens_to_ids(tok) for tok in tokenizer.tokenize(re.sub(r'[ ]+',' ',re.sub(r'[^0-9\.]',' ',a)))]) for a in tqdm(arr,desc="max_numerical_word_rank()")]
    
#=====================================================================

# Features catering to unified(questions and gold solutions)

def to_float(expr):
    try: 
        return float(expr)
    except:
        try:
            val = eval(expr)
            return float()
        except:
            pass
        return str(expr)

def _resolve_expression(lines,num2arg=None,replace_args=False,verbose=False):

    if verbose: 
        print(lines)
        print(num2arg)
    last = lines[-1]["lhs"]
    if len(lines) > 1:
        past = [(idx,l["lhs"],l["rhs"]) for idx, l in enumerate(lines[:-1])][::-1]
        for val in re.findall(r'\d+(?:[\.\,]\d+)?',last):
            for idx, lhs, rhs in past:
                if to_float(rhs) == to_float(val):
                    past_resolved_lhs, _ = _resolve_expression(lines[:idx+1],num2arg)
                    last = last.replace(val,'('+past_resolved_lhs+')')
                    break
    if replace_args:
        for val in re.findall(r'\d+(?:[\.\,]\d+)?',last):
            if val in num2arg:
                last = last.replace(val,num2arg[val])
    return last, lines[-1]["rhs"]

def resolve_expression(ques,gsols,replace_args=True,verbose=False):
    res_exp = []
    for que, gs in zip(ques,gsols):
        if verbose: print(que); print(gs)

        que = que.replace('\n',' ')
        gs = gs.replace('\n',' ')

        _, arg2num = replace_numbers_with_arguments(que)
        num2arg = {v:k for k,v in arg2num.items()}
        lines = [eq.lstrip('+*/').rstrip('+-*/') for eq in re.findall(r'<<([^>>]+)>>',gs)]
        if not len(lines):
            res_exp.append(("N.A.", None))
            continue
        lines = [{hs:eq for hs, eq in zip(("lhs","rhs"),l.strip().split('='))} for l in lines]  # spli by '='
        res_exp.append(_resolve_expression(lines,num2arg=num2arg,replace_args=replace_args,verbose=verbose))
    return res_exp

def IOU(sA,sB):
    return (len(sA.intersection(sB)) + 1) / (len(sA.union(sB)) + 1)

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

def num_expression_feat(ques,gsols):
    qt_arguments = [replace_numbers_with_arguments(q)[1] for q in ques]
    gt_arg_expressions = [eq for (eq,_) in resolve_expression(ques,gsols,replace_args=True)]

    _parameter_usage = []
    _world_knowledge = []
    for i in tqdm(range(len(ques)),total=len(ques),desc="num_expression_feat()"):
        q_args = set(qt_arguments[i].keys())
        gs_usage = set(re.findall(r'arg[A-Z]',gt_arg_expressions[i]) + re.findall(r'\d+(?:[\.\,]\d+)?',gt_arg_expressions[i]))
        _parameter_usage.append(IOU(gs_usage,q_args))
        _world_knowledge.append(len(gs_usage.difference(q_args)))

    return _parameter_usage, _world_knowledge

######################################################################

def extract_question_features(ques, parse_and_ner_data=None):
    pbar = tqdm(desc="extract_question_features()")
    df = {}
    df["Qx_token_length"] = token_length(ques); pbar.update(1)
    df["Qx_sentence_length"] = sentence_length(ques); pbar.update(1)
    df["Qx_word_length"] = word_length(ques); pbar.update(1)
    df["Qx_arg_count"], df["Qx_num_arg_count"], df["Qx_word_arg_count"] = arg_count(ques); pbar.update(1)
    df["Qx_flesch_reading_ease"] = flesch_reading_ease(ques); pbar.update(1)
    df["Qx_flesch_kinkaid_grade"] = flesch_kinkaid_grade(ques); pbar.update(1)
    df["Qx_mean_word_rank"] = mean_word_rank(ques); pbar.update(1)
    df["Qx_mean_numerical_word_rank"] = mean_numerical_word_rank(ques); pbar.update(1)
    if parse_and_ner_data:
        df["Qx_constituency_tree_depth"] = constituency_tree_depth(parse_and_ner_data); pbar.update(1)
        df["Qx_np_count"] = np_count(parse_and_ner_data); pbar.update(1)
        df[f"Qx_prp_count"] = prp_count(parse_and_ner_data); pbar.update(1)
        df["Qx_unique_np_count"] = unique_np_count(parse_and_ner_data); pbar.update(1)
        df["Qx_multi_np_count"] = multi_np_count(parse_and_ner_data); pbar.update(1)
    return df

def extract_gold_solution_features(gsols):
    pbar = tqdm(desc="extract_gold_solution_features()")
    df = {}
    df["Gx_arg_count"], df["Gx_num_arg_count"], df["Gx_word_arg_count"]  = arg_count(gsols); pbar.update(1)
    for operation in "+-*/(":
        df[f"Gx_op\'{operation}\'_count"] = op_count(gsols,operation); pbar.update(1)
    df[f"Gx_op_unique_count"] = unique_op_count(gsols); pbar.update(1)
    df[f"Gx_op_diversity"] = op_diversity(gsols); pbar.update(1)
    df["Gx_mean_numerical_word_rank"] = mean_numerical_word_rank(gsols); pbar.update(1)
    return df

def extract_unified_features(ques,gsols):
    pbar = tqdm(desc="extract_unified_features()")
    df = {}
    df["Gx_parameter_usage"], df["Gx_world_knowledge"] = num_expression_feat(ques,gsols); pbar.update(1)
    return df