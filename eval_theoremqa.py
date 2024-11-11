import re
from latex2sympy2 import latex2sympy
import math
# from __init__ import match_answer
from math import sqrt, sin, cos, log, pi, factorial, exp, e
E = 2.718


def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def clean_units(pred_str: str):
    """Clean the units in the number."""
    def convert_pi_to_number(code_string):
        code_string = code_string.replace('\\pi', 'π')
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r'(?<![\d}])\\?π', '3.14', code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r'(\d)(\\?π)', r'\1*3.14', code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r'\{(\\?π)\}', '3.14', code_string)
        code_string = re.sub(r'\*(\\?π)', '*3.14', code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace('%', '/100')
    pred_str = pred_str.replace('$', '')
    pred_str = pred_str.replace('¥', '')
    pred_str = pred_str.replace('°C', '')
    pred_str = pred_str.replace(' C', '')
    pred_str = pred_str.replace('°', '')
    return pred_str


def number_it(num):
    if isinstance(num, (int, float)):
        return num

    num = clean_units(num)
    try:
        num = str(latex2sympy(num))
    except Exception:
        pass

    if floatify(num) is not None:
        return floatify(num)
    else:
        try:
            num = eval(num)
            if isinstance(num, list) or isinstance(num, tuple):
                num = num[0]
            if floatify(num) is not None:
                return floatify(num)
            else:
                return None
        except Exception:
            return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False,p
        if isinstance(gt, int):
            return round(p) == gt,p
        else:
            return within_eps(pred=p, gt=gt),p
    except Exception:
        return False,p


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False,pred
    elif len(pred) != len(gt):
        return False,pred
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False,pred
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)]),pred


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ['yes', 'true']]):
        pred = 'True'
    elif any([option in pred.lower() for option in ['no', 'false']]):
        pred = 'False'
    elif any([option in pred.lower() for option in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']]):
        pass
    else:
        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split('=')[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r'-?[\d\.]+\s\D+$', pred):
                    pred = pred.split(' ')[0]
                elif re.match(r'-?[\d\.]+\s[^\s]+$', pred):
                    pred = pred.split(' ')[0]
        else:
            # desparate search over the last number
            preds = re.findall(r'-?\d*\.?\d+', pred)
            if(len(preds) >= 1):
                pred = preds[-1]
            else:
                pred = ''

    return pred

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    for ans_marker in ['answer:', "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()
            if response.endswith("\n"):
                response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    for ans_marker in ['be ', "is ", "are ", "=", ": ", "get ", 'be\n', "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker):].strip()
            if response.endswith("\n"):
                response = response[:-2]

    is_matched = is_matched if any([c.isdigit() for c in response]) else False  # answer must have a digit
    # Grade
    return is_matched, response

def answer_clean(direct_answer_trigger_for_fewshot: tuple, pred: str):
    pred = pred.strip('\n')

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split('\n\n')[0]

    # Split the trigger to find the answer.
    # preds = re.split('|'.join(direct_answer_trigger_for_fewshot), pred.lower())
    # if len(preds) > 1:
    #     answer_flag = True
    #     pred = preds[-1]
    # else:
    #     answer_flag = False
    
    answer_flag,pred = match_answer(pred)
    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')

    pred = [extract_theoremqa_answer(pred, answer_flag)]
    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip('.').rstrip('/')

    return pred


def eval_theoremqa(answer: str, groundtruth_str: str, groundtruth_num = None):
    answer = answer_clean(['the answer is:', 'the answer is', 'the answer is','answer:','answers are','answers are:'], answer)
    if groundtruth_str.lower() in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        return groundtruth_str.lower() in answer.lower(),answer.lower()
    elif answer.lower() == groundtruth_str.lower():
        return True,answer.lower()
    elif groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return compare_two_numbers(number_it(answer), groundtruth_num)
        else:
            if answer.startswith('(') and answer.endswith(')'):
                try:
                    answer = list(eval(answer))
                    answer = [number_it(a) for a in answer]
                except Exception as e:
                    return False,answer
                return compare_two_list(answer, groundtruth_num)
            else:
                return False,answer
    else:
        return False,answer