import json

def judge_average(file_path):
    total_score = 0
    count = 0

    with open(file_path, "r") as file:
        for line in file:
            try:
                data = json.loads(line)
                score = 1 if data["accuracy"] else 0
                total_score += score
                count += 1
            except json.JSONDecodeError:
                continue

    average_score = total_score / count if count > 0 else 0
    return average_score

stage_mapping = {
    "1": "none",
    "2": "pre-retrieval",
    "3": "cot",
    "4": "rig",
    "5": "pre-retrieval+cot",
    "6": "cot+rig",
    "7": "pre-retrieval+rig",
    "8": "pre-retrieval+cot+rig"
}

user_input = input("Select stage (1: none, 2: pre-retrieval, 3: cot, 4: rig, 5: pre-retrieval+cot, 6: cot+rig, 7: pre-retrieval+rig, 8: pre-retrieval+cot+rig): ")
if user_input not in stage_mapping:
    raise ValueError("Invalid stage selected. Please input a number between 1 and 8")

stage = stage_mapping[user_input]

file_path = f'{stage}/judge_results.jsonl'
judge_average = judge_average(file_path)
print(f"Average score of LLM as judge for {stage}: {judge_average}")