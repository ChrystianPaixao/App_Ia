# explore.py
import json
import os

with open('dataset_small/metadata.json', 'r') as f:
    data = json.load(f)

print("Tipo do dado:", type(data))
if isinstance(data, list):
    print("Número de itens:", len(data))
    if len(data) > 0:
        print("Primeiro item:", json.dumps(data[0], indent=2))
else:
    print("Chaves disponíveis:", data.keys())
    print("Exemplo:", json.dumps(list(data.values())[0], indent=2))