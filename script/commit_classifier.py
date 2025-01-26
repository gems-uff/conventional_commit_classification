import json
from ollama import chat
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import (
    DATA_PATH,
)

input_file = DATA_PATH / "maven/mockito/mockito_v5.15.2.json"  
output_file = "commits.json"  # Arquivo de saída com as classificações

def classify_commit(commit_message):
    # Prompt para o modelo Llama
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a specialized assistant for classifying commit messages into Semantic Versioning categories: "Patch", "Minor", or "Major". '
                'Here are the rules:\n'
                '- "Patch": Fixes bugs, refactors code without changing functionality, or makes small optimizations without affecting the API.\n'
                '- "Minor": Adds new features or functionalities in a backward-compatible manner.\n'
                '- "Major": Introduces breaking changes that are not backward-compatible or significantly alters existing functionality.\n'
                'Respond with the following format:\n'
                '1. Category: [Patch/Minor/Major]\n'
                '2. Reason: [a brief reason for your choice..]\n'
                'Example response:\n'
                'Category: Patch\n'
                'Reason: Fixes a bug in the calculation.\n'
            )
        },
        {
            'role': 'user',
            'content': commit_message
        }
    ]

    # Chamada ao modelo
    response = chat(model='llama3.2', messages=messages)
    classification = response['message']['content']

    # Processa a resposta para extrair as partes
    lines = classification.split("\n")
    category = next((line.split(": ")[1] for line in lines if line.startswith("Category:")), "Unknown")
    reason = next((line.split(": ")[1] for line in lines if line.startswith("Reason:")), "No reason provided")

    return category, reason

# Processamento do arquivo JSON
def process_commits(input_path, output_path):
    with open(input_path, 'r') as infile:
        data = json.load(infile)

    output = []
    for commit in data.get('commits_info', []):
        hash_value = commit.get('hash')
        message = commit.get('message')
        
        if message:
            # Classifica o commit
            category, reason = classify_commit(message)
            
            # Adiciona ao resultado
            output.append({
                "hash": hash_value,
                "bump": category,
                "reason": reason,
            })

    # Escreve o resultado em um novo arquivo JSON
    with open(output_path, 'w') as outfile:
        json.dump(output, outfile, indent=4)

# Executa o processamento
process_commits(input_file, output_file)
print(f"Classification completed. Results saved to {output_file}.")