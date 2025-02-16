import time
import json
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SYS_PROMPT = """You are a helpful assistant to jusge whether the model's final response (might be word, phrase or sentence) and the given correct answer is same in value.
- If their intrinsic numerical value of the answer is not equal, please mark it as wrong.
- If they are just expressed in different format or wording or unit, but have the same main value, please mark it as correct.

Example:
- Model response: The final answer should be 4.123
- Ground truth: \sqrt{17}
- Judgment: correct

- Model response: 0.2687
- Ground truth: \frac{pi}{9}
- Judgment: wrong

- Model response: 40%
- Ground truth: 40
- Judgment: correct

- Model response: Therefore, the speed of the car is 25 miles per hour
- Ground truth: 25
- Judgment: correct

- Model response: The temperature of the metal ay noon will be 10938.893 T
- Ground truth: 1.983e4
- Judgment: wrong

- Model response: $1000.00
- Ground truth: 1000
- Judgment: correct"""

USER_PROMPT = """- Model response: <pd>
- Ground truth: <gt>
- Judgment: """

if os.path.exists("../secret.json"):
    client = OpenAI(
        api_key=json.load(open("../secret.json"))["api_key"],
        base_url=json.load(open("../secret.json"))["base_url"]
    )
else:
    client = OpenAI(api_key="sk-...")


def form_messages(msg: str, system_prompt: str = ""):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': msg}
    ]
    return messages


def gpt_chatcompletion(messages, model="gpt-4o-mini"):
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.000001,
                n=1,
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")

judgment = {
    "correct": 0,
    "wrong": 0
}

def main(data, answered_data, hash_tab, save_path, log, model):
    try:
        if data['task'] in hash_tab:
            return
        
        task = data['task']
        pd = data['predict'][-1]['reasoning'].replace("\n", " ").strip()
        gt = data['ground_truth'].split("### Final Response")[1].strip()
        
        if pd.lower() == gt.lower():
            judgment["correct"] += 1
            data['judge'] = "correct"
            answered_data.append(data)
            log["success"] += 1
            return
        
        print("\n======================= Question ========================\n")
        print(task)
        user_prompt = USER_PROMPT.replace("<pd>", pd).replace("<gt>", gt)
        system_prompt = SYS_PROMPT
        messages = form_messages(user_prompt, system_prompt)
        response = gpt_chatcompletion(messages, model=model).strip()
        print("\n======================= Response ========================\n")
        print(response)
        
        if response == "correct":
            judgment["correct"] += 1
        elif response == "wrong":
            judgment["wrong"] += 1  
        else:
            assert False, "Unknown judgment: " + response
        
        data['judge'] = response
        answered_data.append(data)
        
        if len(answered_data) % 50 == 0:
            with open(save_path, 'w') as f:
                f.write(json.dumps(answered_data, indent=2))
        
        log["success"] += 1
    except:
        log["fail"] += 1
    
    print(log)
    print(judgment)
    return


if __name__ == '__main__':
    data_path = f"PATH/TO/INFERENCE/DATA.json"
    all_data = json.load(open(data_path))
    
    save_path = data_path.replace(".json", "_judge.json")
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            answered_data = json.load(f)
        hash_tab = []
        for data in answered_data:
            hash_tab.append(data['task'])
    else:
        answered_data = []
        hash_tab = []
    
    print(f"Existing data: {len(answered_data)}")
    
    model = "gpt-4o"
    log = {"success": 0, "fail": 0}
    
    # Run conversations in parallel
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(main, data, answered_data, hash_tab, save_path, log, model) for data in tqdm(all_data)]
        for future in futures:
            future.result()
    
    with open(save_path, 'w') as f:
        f.write(json.dumps(answered_data, indent=2))
    
    print(log)
    print(judgment)
    
    final_cw_dict = {"correct": 0, "wrong": 0}
    for data in answered_data:
        if data['judge'] == "correct":
            final_cw_dict["correct"] += 1
        elif data['judge'] == "wrong":
            final_cw_dict["wrong"] += 1
    correct_rate = round(final_cw_dict["correct"] / (final_cw_dict["correct"] + final_cw_dict["wrong"]), 4)
    print(f"Correct rate: {correct_rate}")
    print(final_cw_dict)
