import os
import time
import requests
import json


time.sleep(3)
url = "http://127.0.0.1:12345/execute"
file_path = os.path.join('inputs', 'input.json')
with open(file_path, 'r') as file:
    data = json.load(file)
response = requests.post(url, json=data)
print(response.json())