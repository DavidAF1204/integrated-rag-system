from llm_config import *
from cot import llm_adapter
from typing import Sequence
import json

def generate_answer(contexts: Sequence[str], query: str, use_cot: bool = False) -> str:
    if use_cot:
        return llm_adapter(contexts, query)
    
    system_prompt = f"""Context information is below.
---------------------
{contexts}
---------------------"""

    user_prompt = f"""Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: """

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

def process_test_cases(use_pre_retrieval: bool = False, use_cot: bool = False):
    generated = 0
    
    input_path = "../results/contexts_with_pre-retrieval.jsonl" if use_pre_retrieval else "../results/contexts_without_pre-retrieval.jsonl"
    
    if use_pre_retrieval:
        output_dir = "pre-retrieval+cot" if use_cot else "pre-retrieval"
    else:
        output_dir = "cot" if use_cot else "none"
    
    output_path = f"../results/{output_dir}/test_cases.jsonl"

    with open(input_path, 'r') as contexts_file, \
         open(output_path, 'w') as answers_file:
        
        lines = contexts_file.readlines()

        for line in lines:
            item = json.loads(line)
            
            answer_item = {
                'id': item['id'],
                'query': item['query'],
                'ground_truth': item['ground_truth'],
                'answer': generate_answer(item['contexts'], item['query'], use_cot)
            }
            
            generated += 1
            cot_status = "with CoT" if use_cot else "without CoT"
            print(f"Generated answers {cot_status} for {generated} test cases")

            answers_file.write(json.dumps(answer_item) + '\n')
            answers_file.flush()
    
    print(f"Answers generation completed. Results stored in {output_path}")

use_pre_retrieval = input("Used pre-retrieval for contexts generation? (T/F): ").upper() == 'T'
use_cot = input("Use Chain of Thought reasoning for answers generation? (T/F): ").upper() == 'T'
process_test_cases(use_pre_retrieval, use_cot)