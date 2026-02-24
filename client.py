import requests
import json

def test_stream(prompt):
    url = "http://localhost:11434/generate/stream"
    payload = {
        "prompt": prompt,
        "max_length": 200,
        "temperature": 0.7
    }

    print(f"--- Prompt: {prompt} ---\n")
    
    # stream=True is key here to catch the generator output
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code == 200:
            print("Response: ", end="", flush=True)
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
            print("\n\n--- Done ---")
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_stream("Write a Python function to sort a list of integers.")
