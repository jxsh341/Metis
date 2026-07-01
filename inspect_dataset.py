import json

with open('data/metis/metis_dataset.json') as f:
    data = json.load(f)

print(f'Total examples: {len(data)}')
print(f'Domains: {set(d["domain"] for d in data)}')
print()
print('=== SAMPLE EXAMPLE ===')
ex = data[2]
print(f'PROBLEM: {ex["problem"][:150]}')
print()
print(f'STAGE 1: {ex["initial_hypothesis"][:200]}')
print()
print(f'STAGE 2: {ex["assumption_audit"][:200]}')
print()
print(f'STAGE 3: {ex["self_critique"][:200]}')
print()
print(f'STAGE 4: {ex["revised_reasoning"][:200]}')
print()
print(f'STAGE 5: {ex["grounded_conclusion"][:200]}')