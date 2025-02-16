import json
import time
import os
from openai import OpenAI

# Define the system prompt
SYS_PROMPT = """You are given a very general task. Now, you should pretend to be a user and give answer to a related query. You should provide a response to the query to show your preferences or requirements.

Please directly provide a concise and coherent response."""

# Load the API key
if os.path.exists("../secret.json"):
    client = OpenAI(
        api_key=json.load(open("../secret.json"))["api_key"],
        base_url=json.load(open("../secret.json"))["base_url"]
    )
else:
    client = OpenAI(api_key="sk-...")


def form_messages(task, query):
    """
    Format the messages for GPT interaction.
    """
    user_prompt = f"### Task{task}\n\n### Query\n{query}\n\n### Response\n"
    messages = [
        {'role': 'system', 'content': SYS_PROMPT},
        {'role': 'user', 'content': user_prompt}
    ]
    return messages


def gpt_chatcompletion(messages, model="gpt-4o"):
    """
    Perform GPT chat completion with retries.
    """
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                n=1,
            )
            content = response.choices[0].message.content
            return content.replace("### Response", "").strip()
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")


def simulate_user_response(task, query, model="gpt-4o"):
    """
    Simulate a user response to a query within a given task.
    
    Parameters:
    - task: The context or task description for the reasoning process.
    - query: The reasoning query chain to which GPT will respond.
    - model: The GPT model to use (default is "gpt-4o-mini").
    
    Returns:
    - The simulated user response.
    """
    try:
        # Format the messages
        messages = form_messages(task, query)
        # Call GPT and get the response
        response = gpt_chatcompletion(messages, model=model)
        return response
    except Exception as e:
        print(f"Error in simulate_user_response: {e}")
        return None
