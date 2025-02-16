import os
import re
import subprocess

def execute_code(code_text, file_name="../env/tmp.py"):
    # Extract code between the markdown-style triple backticks
    code_match = re.search(r"```python\n(.*?)```", code_text, re.DOTALL)
    if not code_match:
        return "Error: No valid Python code block found."

    # Get the code from the match
    code = code_match.group(1)
    
    # Save the code to the specified file
    try:
        with open(file_name, "w") as file:
            file.write(code)
    except Exception as e:
        return f"Error: Could not write to file. {e}"

    # Execute the file and capture the output or errors
    try:
        result = subprocess.run(
            ["python", file_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=20  # Set timeout for 20 seconds
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        output = "Error: Execution time exceeded the 20-second limit."
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr.strip()}"
    finally:
        # Delete the file if it exists
        if os.path.exists(file_name):
            os.remove(file_name)

    if len(output) > 256:
        output = output[:128] + "..." + output[-128:]
    
    return output