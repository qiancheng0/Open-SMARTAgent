import json
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
from cprint import *


def preprocess_dataset(data_path, max_num=-1, start_id=0, method="llama"):
    """
    Load and preprocess the dataset by applying the chat template.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = []
    max_num = len(data) if max_num == -1 else max_num
    
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


def inference(args):
    """
    Perform inference using the vLLM API.
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
    
    for example in tqdm(dataset):
        input_messages = example["input"]
        ground_truth = example["ground_truth"]
        task = example["task"]
        
        if task in existing_tasks:
            continue
        
        steps = []
        step_time = 0
        
        while True:
            try:
                step_time += 1
                if step_time > 1:
                    steps.append({
                        "name": "Reasoning",
                        "type": "normal",
                        "tool_name": None,
                        "reasoning": assistant_output
                    })
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
                    })
                    with open(args.save_path, "w") as f:
                        json.dump(results, f, indent=2)
                    break
                
                result = llm.chat(input_messages, sampling_params=sampling_params)
                assistant_output = result[0].outputs[0].text.strip()
                
                cprint.info("\n\n", "+" * 10, "Round Response", "+" * 10)
                print(assistant_output)
                
                if "### Final Response" not in assistant_output:
                    continue
                reasoning = assistant_output.split("### Final Response")[0].strip()
                final_output = assistant_output.split("### Final Response")[1].strip()
                
                steps.append({
                    "name": "Reasoning",
                    "type": "normal",
                    "tool_name": None,
                    "reasoning": reasoning
                })
                steps.append({
                    "name": "Final Response",
                    "type": "normal",
                    "tool_name": None,
                    "reasoning": final_output
                })
                
                log["success"] += 1
                cprint.info("\n\n", "+" * 10, "Ground Truth", "+" * 10)
                print(ground_truth)
                
                results.append({
                    "task": task,
                    "predict": steps,
                    "ground_truth": ground_truth
                })
                
                with open(args.save_path, "w") as f:
                    json.dump(results, f, indent=2)
                
                print(log)
                break
            
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
