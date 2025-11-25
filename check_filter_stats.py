import json

with open('outputs/filtered_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stats = {'FORCE_ANONYMIZE': 0, 'FORCE_IGNORE': 0, 'ESCALATE_TO_LLM': 0}
for entry in data:
    decision = entry.get('decision', 'UNKNOWN')
    stats[decision] = stats.get(decision, 0) + 1

print(f"FORCE_ANONYMIZE: {stats['FORCE_ANONYMIZE']}")
print(f"FORCE_IGNORE: {stats['FORCE_IGNORE']}")  
print(f"ESCALATE_TO_LLM: {stats['ESCALATE_TO_LLM']}")
print(f"Total: {len(data)}")
reduction = (1 - stats['ESCALATE_TO_LLM']/len(data)) * 100 if len(data) > 0 else 0
print(f"Reducción LLM: {reduction:.1f}%")
