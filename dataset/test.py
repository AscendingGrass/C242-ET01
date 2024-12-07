import json

with open('dataset/medium_70.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(len(data))