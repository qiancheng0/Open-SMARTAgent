import json
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
from cprint import *

from .utils_serper import search_serper
from .utils_askuser import simulate_user_response
from .utils_code import execute_code


def extract_first_parentheses_content(text):
    start = text.find('(')
    if start == -1:
        return None
    stack = []
    content = []
    for i in range(start + 1, len(text)):
        char = text[i]
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return ''.join(content).strip()
            stack.pop()
        content.append(char)
    return None


def find_earliest_string(text):
    substrings = {
        "AskUser": "AskUser(",
        "Search": "Search(",
        "Code": "```python"
    }
    occurrences = {key: text.find(value) for key, value in substrings.items()}
    valid_occurrences = {key: idx for key, idx in occurrences.items() if idx != -1}
    if not valid_occurrences:
        return -1, None
    
    earliest_key = min(valid_occurrences, key=valid_occurrences.get)
    earliest_index = valid_occurrences[earliest_key]
    return earliest_index, earliest_key


def parse_steps(text):
    if "### Output Guidelines" in text:
        text = text.split("### Output Guidelines")[0].strip()
    if "** Input **" in text:
        text = text.replace("** Input **", "")
    if "** Output **" in text:
        text = text.replace("** Output **", "")
    if "### Reasoning Steps" in text:
        text = text.replace("### Reasoning Steps", "").strip()
    if "### Continue your reasoning" in text:
        text = text.replace("### Continue your reasoning", "").strip()
    text = text.strip()
    
    earliest_index, tool_name = find_earliest_string(text)
    
    if tool_name is not None:
        if tool_name == "AskUser":
            before_tool = text[:earliest_index].strip()
            tool_content = extract_first_parentheses_content(text[earliest_index:]).strip()
        if tool_name == "Search":
            before_tool = text[:earliest_index].strip()
            tool_content = extract_first_parentheses_content(text[earliest_index:]).strip()
        if tool_name == "Code":
            before_tool = text[:earliest_index].strip()
            tool_content = text[earliest_index:].split("```python")[1].split("```")[0].strip()
            tool_content = f"```python\n{tool_content}\n```"
        return [
            {
                "name": "Reasoning Step",
                "type": "normal",
                "tool_name": None,
                "reasoning": before_tool
            },
            {
                "name": "Reasoning Step",
                "type": "tool",
                "tool_name": tool_name,
                "reasoning": tool_content
            }
        ]
    
    if "Final Answer:" in text:
        reasoning_before = text.split("Final Answer:")[0].strip()
        reasoning_after = text.split("Final Answer:")[1].strip()
        return [
            {
                "name": "Reasoning Step",
                "type": "normal",
                "tool_name": None,
                "reasoning": reasoning_before
            },
            {
                "name": "Final Response",
                "type": "normal",
                "tool_name": None,
                "reasoning": reasoning_after
            }
        ]
        
    return [
        {
            "name": "Reasoning Step",
            "type": "normal",
            "tool_name": None,
            "reasoning": text
        }
    ]


def format_steps(steps):
    results = []
    for idx, step in enumerate(steps):
        if step["type"] == "normal":
            results.append(step["reasoning"].strip())
        elif step["tool_name"] == "AskUser" or step["tool_name"] == "Search":
            results.append(f"{step['tool_name']}({step['reasoning']})")
        elif step["tool_name"] == "Code":
            results.append(step['reasoning'])
        if "output" in step:
            output = step["output"]
            results.append(f"- Tool Output: {output}")
    return "\n".join(results)



def preprocess_dataset(data_path, max_num, start_id, method):
    """
    Load and preprocess the dataset by applying the chat template.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    max_num = len(data) if max_num == -1 else max_num
    dataset = []
    for d in data[start_id:]:
        task = d["input"].split("### Task")[1].split("###")[0].strip()
        
        if method == "mistral":
            messages = [
                {
                    "role": "user",
                    "content": d["instruction"].strip() + "\n\n" + d["input"].strip()
                }
            ]
        elif method == "llama":
            messages = [
                {
                    "role": "system",
                    "content": d["instruction"],
                },
                {
                    "role": "user",
                    "content": d["input"],
                }
            ]
        dataset.append({"input": messages, "ground_truth": d["output"], "task": task})
        
        if len(dataset) >= max_num:
            break
    
    print(f"Length of data: {len(dataset)}")
    return dataset


def parse_code_content(text):
    code = text.split("```python")[1].split("```")[0].strip()
    code_lines = code.split("\n")
    
    new_lines = []
    for line in code_lines:
        if line.strip().startswith("#"):
            continue
        if line.startswith("print("):
            continue
        new_lines.append(line)
    
    new_code = "\n".join(new_lines)
    return new_code


def inference(args):
    """
    Perform inference using the pipeline API.
    """
    # Load the model using vLLM
    print("Loading model with vLLM...")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=int(os.getenv("WORLD_SIZE", 1)),
        gpu_memory_utilization=0.92,
        max_model_len=args.max_seq_length,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_seq_length,
        temperature=0.000001,
    )

    # Preprocess dataset
    print("Loading and preprocessing dataset...")
    dataset = preprocess_dataset(args.data_path, max_num=args.max_test_num, start_id=args.test_start_id, method=args.method)
    
    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path, "r", encoding="utf-8"))
        existing_tasks = [r["task"] for r in results]
    else:
        results = []
        existing_tasks = []
    
    print(f"Length of existing data: {len(results)}")
    
    # Perform inference
    print("Starting inference...")
    log = {"fail": 0, "success": 0}

    example_count = 0
    
    for example in tqdm(dataset):
        input_messages = example["input"]
        ground_truth = example["ground_truth"]
        task = example["task"]
        
        example_count += 1
        code_file = str(example_count) + "_" + task[:3] + ".py"
        
        if task in existing_tasks:
            continue
        
        steps = []
        raw = []
        step_time = 0
        
        all_previous_code = ""
        
        while True:
            try:
                step_time += 1
                if step_time > 10:
                    steps.append({
                        "name": "Final Response",
                        "type": "normal",
                        "tool_name": None,
                        "reasoning": "Still do not get an answer after exceeding maximum step time! Please judge the answer for this question as wrong."
                    })
                    results.append({
                        "task": task,
                        "predict": steps,
                        "ground_truth": ground_truth,
                        "raw": raw
                    })
                    with open(args.save_path, "w") as f:
                        json.dump(results, f, indent=2)
                    break
                

                result = llm.chat(input_messages, sampling_params)
                assistant_output = result[0].outputs[0].text.strip()
                
                raw.append(assistant_output)
                
                cprint.info("\n\n", "+" * 10, "Round Response", "+" * 10)
                print(assistant_output)
                
                new_steps = parse_steps(assistant_output)
                
                if new_steps[-1]["name"] == "Final Response":
                    log["success"] += 1
                    cprint.info("\n\n", "+" * 10, "Ground Truth", "+" * 10)
                    print(ground_truth)
                    
                    steps.extend(new_steps)
                    
                    results.append({
                        "task": task,
                        "predict": steps,
                        "ground_truth": ground_truth,
                        "raw": raw
                    })
                    
                    with open(args.save_path, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    print(log)
                    break
                
                if new_steps[-1]["type"] == "tool":
                    tool_name = new_steps[-1]["tool_name"]
                    if tool_name == "AskUser":
                        cprint.info("AskUser tool detected")
                        response = simulate_user_response(task, new_steps[-1]["reasoning"])
                        new_steps[-1]["output"] = response
                    elif tool_name == "Search":
                        cprint.ok("Search tool detected")
                        if "intention" in args.data_path:
                            link = True
                        else:
                            link = False
                        response = search_serper(new_steps[-1]["reasoning"], link=link, num=3)
                        new_steps[-1]["output"] = response
                    elif tool_name == "Code":
                        cprint.warn("Code tool detected")
                        
                        code_content = new_steps[-1]["reasoning"].split("```python")[1].split("```")[0].strip()
                        code_content = f"```python\n{all_previous_code}\n{code_content}\n```"
                        
                        print(" ====== Code Content ====== ")
                        print(code_content)
                        
                        response = execute_code(code_content, "./env/" + code_file)
                        
                        if not response.startswith("Error"):
                            new_code = parse_code_content(new_steps[-1]["reasoning"])
                            all_previous_code += new_code + "\n"
                        
                        new_steps[-1]["output"] = response
                    else:
                        assert False, "Unknown tool name"
                    cprint.info("\n\n", "+" * 10, f"Tool {tool_name} Response", "+" * 10)
                    print(response)
                
                input_messages[-1]["content"] = input_messages[-1]["content"].strip() + "\n" + format_steps(new_steps).strip() + "\n\n### Continue your reasoning\n"
                steps.extend(new_steps)
            
            except Exception as e:
                log["fail"] += 1
                print(e)
                break
        

def initialize():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the inference results")
    parser.add_argument("--test_start_id", type=int, default=0, help="The start id for testing")
    parser.add_argument("--max_test_num", type=int, default=-1, help="The max number of instances to test")
    parser.add_argument("--method", type=str, default="llama", help="text or message")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = initialize()
    inference(args)
