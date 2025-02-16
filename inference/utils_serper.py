import http.client
import json
import time

api_key = json.load(open("../secret.json"))["serper_key"]

conn = http.client.HTTPSConnection("google.serper.dev")
headers = {
  'X-API-KEY': api_key,
  'Content-Type': 'application/json'
}

def search_serper(query, link=False, num=10):
    payload = json.dumps({
        "q": query,
        "tbs": "qdr:y"
    })

    try_time = 0
    while True:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")
        data = json.loads(data)
        if try_time > 10:
            return "Search Error: Timeout"
        if data["organic"] != []:
            break
        try_time += 1
        time.sleep(5)
        
    try:
        output = ""
        index = 1
        answer_box = data.get("answerBox", "")
        if answer_box:
            if link:
                if 'title' in answer_box and 'link' in answer_box and 'snippet' in answer_box:
                    output += f"{str(index)}. {answer_box['title']}\n- Link: {answer_box['link']}\n- Snippet: {answer_box['snippet']}\n"
                    index += 1
            else:
                if 'title' in answer_box and 'date' in answer_box and 'snippet' in answer_box:
                    output += f"{str(index)}. {answer_box['title']}\n- Date: {answer_box['date']}\n- Snippet: {answer_box['snippet']}\n"
                    index += 1
        
        if index > num:
            return output.strip()
        
        for item in data.get("organic", []):
            if link:
                if 'title' in item and 'link' in item and 'snippet' in item:
                    output += f"{str(index)}. {item['title']}\n- Link: {item['link']}\n- Snippet: {item['snippet']}\n"
                    index += 1
            else:
                if 'title' in item and 'date' in item and 'snippet' in item:
                    output += f"{str(index)}. {item['title']}\n- Date: {item['date']}\n- Snippet: {item['snippet']}\n"
                    index += 1
            if index > num:
                return output.strip()
        
        return output.strip()
    
    except Exception as e:
        error = f"Search Error: {e}"
        print(error)
        return error

