import time
import json
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SYS_PROMPT_MISSING = """You are a helplful assistant to help me judge if a given aspect of detail is covered by the questions provided. Please first give a brief thought afte '- Thought'. If it is covered by at least one of the question, please respond 'Yes' in the after '- Judgment', other wise respond with 'No'.

- Aspect of Detail: Specific email address to investigate
- Questions:
    1. Ask about the email address: Could you please provide the email address you want to check for spam activities?
    2. Ask about the purpose of the check: What is the purpose of checking this email address for spam activities? Are you a recipient, sender, or investigating a specific issue?
    3. Ask about preferred method of checking: What is your preferred method of checking the email address for spam activities? Do you have a specific tool or service in mind?
- Thought: The specific email address to investigate is covered by the first questions, as it asks for the email address to check for spam activities.
- Judgment: Yes

- Aspect of Detail: How skilled the user is in using Linux
- Questions:
    1. Ask about preferred monitoring tool: What is your preferred tool for monitoring CPU and memory usage, such as `top`, `htop`, `vmstat`, `sar`, or `nmon`?
    2. Ask about preferred output format: What is your preferred output format for the monitoring data? Do you want a simple text summary or a detailed graphical representation?
    3. Ask about time interval: What time interval would you like for the monitoring updates? For example, every second, minute, or hour?
- Thought: How skilled the user is in using Linux is not covered by the questions provided. It is not related to the preferred monitoring tool, output format, or time interval.
- Judgment: No
"""

USER_PROMPT_MISSING = """- Aspect of Detail: <missing_detail>
- Questions:
<questions>"""

SYS_PROMPT_SUMMARY = """You are a helplful assistant to help me judge if a user's given preference information is properly addressed in the summary provided. Please first give a brief thought afte '- Thought'. If it is covered by the summary, please respond 'Yes' in the after '- Judgment', other wise respond with 'No'.

- Summary: To check if the email address 'example@example.com' is associated with spam activities, you can use online tools like Spamhaus and MXToolbox, which provide comprehensive blacklists and reputation checks.
1. **Spamhaus**: You can visit the Spamhaus website to see if 'example@example.com' is listed in their blacklists. Spamhaus is a well-known service that tracks spamming activity and maintains various blacklists. You can check their resources on how to handle bounced emails and understand their blacklist system. [Spamhaus](https://www.spamhaus.org/)
2. **MXToolbox**: This tool offers a blacklist checker that scans multiple email blacklists to monitor mail server reputation. It's a popular choice for businesses and individuals to check their email security. You can explore more about blacklists and their usage on the MXToolbox Blog. [MXToolbox Blog](https://blog.mxtoolbox.com/tag/blacklists/)
These tools will help you assess the reputation of the email address and identify any potential spam-related issues.
- User preference: I prefer using online tools like Spamhaus or MXToolbox to check if an email address is associated with spam activities. These services provide comprehensive blacklists and reputation checks that help identify any suspicious activity linked to the email address.
- Thought: The user's preference for using online tools like Spamhaus and MXToolbox to check for spam activities is properly addressed in the summary. The summary provides detailed information about these tools and how they can be used to monitor mail server reputation and identify potential spam-related issues.
- Judgment: Yes

- Summary: To monitor CPU and memory usage continuously with updates every second, you can use the `vmstat` command. Here is the command you can use:
```bash
vmstat 1 10
```
This command will display a simple text summary of CPU and memory usage every second for 10 seconds, providing real-time updates that align with your preferences.
- User preference: I would like the command to be not very complex as I am new to Linux and not skilled in using command-line tools.
- Thought: The user's preference for a command siimply for new Linux user is not explicitly addressed in the summary. The summary provides the `vmstat` command for monitoring CPU and memory usage, but it does not mention the complexity of the command or how it aligns with the user's skill level.
- Judgment: No
"""

USER_PROMPT_SUMMARY = """- Summary: <summary>
- User preference: <user_preference>
"""

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


def gpt_chatcompletion(messages, model="gpt-4o"):
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


def main(data, answered_data, hash_tab, save_path, log, model):
    try:
        if data['task'] in hash_tab:
            return
        
        missing_details = data['missing_details']
        predict = data['predict']
        questions = ""
        count = 1
        for p in predict:
            if p["tool_name"] != "AskUser":
                continue
            n = p["name"]
            q = p["reasoning"]
            questions += f"    {count}. {n}: {q}\n"
            count += 1
        questions = questions.strip()
        
        missing_results = []
        
        zero_question_flag = False
        if questions == "":
            zero_question_flag = True
                
        for missing_detail in missing_details:
            description = missing_detail['description']
            importance = missing_detail['importance']
            
            if zero_question_flag:
                missing_results.append({"description": description, "importance": importance, "thought": "Not applicable, no questions provided", "judgment": "No"})
                continue
            
            user_prompt = USER_PROMPT_MISSING.replace("<missing_detail>", description).replace("<questions>", questions)
            system_prompt = SYS_PROMPT_MISSING
            messages = form_messages(user_prompt, system_prompt)
            response = gpt_chatcompletion(messages, model=model).strip()
            print("\n======================= Response ========================\n")
            print(response)
            thought = response.split("- Thought:")[1].split("- Judgment:")[0].strip()
            judgment = response.split("- Judgment:")[1].strip()
            missing_results.append({"description": description, "importance": importance, "thought": thought, "judgment": judgment})
        
        data['missing_results'] = missing_results
        
        
        summary = data["predict"][-1]["reasoning"]
        summary_results = []
        
        for p in predict:
            if p["tool_name"] != "AskUser":
                continue
            user_preference = p["output"].split("### Response")[-1].strip()
            if summary.startswith("Still do not"):
                summary_results.append({"user_preference": user_preference, "thought": "Not applicable, response unfinished", "judgment": "No"})
                continue
            
            user_prompt = USER_PROMPT_SUMMARY.replace("<summary>", summary).replace("<user_preference>", user_preference)
            system_prompt = SYS_PROMPT_SUMMARY
            messages = form_messages(user_prompt, system_prompt)
            response = gpt_chatcompletion(messages, model=model).strip()
            print("\n======================= Response ========================\n")
            print(response)
            thought = response.split("- Thought:")[1].split("- Judgment:")[0].strip()
            judgment = response.split("- Judgment:")[1].strip()
            summary_results.append({"user_preference": user_preference, "thought": thought, "judgment": judgment})
        
        data['summary_results'] = summary_results
        
        answered_data.append(data)
        
        if len(answered_data) % 20 == 0:
            with open(save_path, 'w') as f:
                f.write(json.dumps(answered_data, indent=2))
        
        log["success"] += 1
    
    except Exception as e:
        print(f"Error: {e}")
        log["fail"] += 1
    
    print(log)
    return


if __name__ == '__main__':
    ref_path = "../data_inference/domain_intention_smart.json"
    ref_data = {}
    for data in json.load(open(ref_path)):
        task = data["input"].split("### Task")[1].split("###")[0].strip()
        missing_details = data["missing_details"]
        ref_data[task] = missing_details
        
    data_path = f"PATH/TO/INFERENCE/DATA.json"
    all_data = json.load(open(data_path))
    
    for data in all_data:
        task = data['task']
        missing_details = ref_data[task]
        assert missing_details != [], f"Missing details not found for task: {task}"
        data['missing_details'] = missing_details        
        
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
    
    missing_lv3_recovers = []
    missing_lv2_recovers = []
    missing_lv1_recovers = []
    summary_successes = []
    for data in answered_data:
        missing_results = data['missing_results']
        summary_results = data['summary_results']
        
        missing_lv3_recover = []
        missing_lv2_recover = []
        missing_lv1_recover = []
        for r in missing_results:
            if r["importance"] == "3":
                missing_lv3_recover.append(1) if r["judgment"] == "Yes" else missing_lv3_recover.append(0)
            elif r["importance"] == "2":
                missing_lv2_recover.append(1) if r["judgment"] == "Yes" else missing_lv2_recover.append(0)
            elif r["importance"] == "1":
                missing_lv1_recover.append(1) if r["judgment"] == "Yes" else missing_lv1_recover.append(0)
        if len(missing_lv3_recover) != 0:
            missing_lv3_recovers.append(sum(missing_lv3_recover) / len(missing_lv3_recover))
        if len(missing_lv2_recover) != 0:
            missing_lv2_recovers.append(sum(missing_lv2_recover) / len(missing_lv2_recover))
        if len(missing_lv1_recover) != 0:
            missing_lv1_recovers.append(sum(missing_lv1_recover) / len(missing_lv1_recover))
        
        summray_success = []
        for r in summary_results:
            summray_success.append(1) if r["judgment"] == "Yes" else summray_success.append(0)
        if len(summray_success) != 0:
            summary_successes.append(sum(summray_success) / len(summray_success))
    
    print(f"Missing Level 3 Recovery: {round(sum(missing_lv3_recovers) / len(missing_lv3_recovers) * 100, 2)}")
    print(f"Missing Level 2 Recovery: {round(sum(missing_lv2_recovers) / len(missing_lv2_recovers) * 100, 2)}")
    print(f"Missing Level 1 Recovery: {round(sum(missing_lv1_recovers) / len(missing_lv1_recovers) * 100, 2)}")
    print(f"Summary Success: {round(sum(summary_successes) / len(summary_successes) * 100, 2)}")    
