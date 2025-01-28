from llm_config import *
from typing import Sequence
import json
import chromadb
import chromadb.utils.embedding_functions
import torch
import time

def generate_detail_queries(answer: str) -> list:
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            prompt = f"""Given this answer, extract key factual statements and generate specific queries to verify each statement. 
If the answer indicates that information is not available or cannot be found in the context, return an empty list [].
Each detail must be a complete statement with a subject and predicate, not just isolated facts like names, dates, or numbers.

Answer: {answer}
Output format: JSON list of dictionaries with 'detail' and 'generated_query' keys. Return an empty list [] if:
1. No complete statements can be extracted, OR
2. The answer indicates information is not available/found in the context (e.g., "The information is not available", "I cannot find this information", "The context doesn't provide this information", etc.)

Example:
Answer: "Yes, the use of renewables has been increasing significantly in the world. Renewable energy sources now provide over 12% of global energy consumption, up from 6% in 2000."
[
    {{
        "detail": "Renewable energy sources now provide over 12% of global energy consumption",
        "generated_query": "What percentage of global energy comes from renewables?"
    }},
    {{
        "detail": "Renewable energy sources provide 6% of global energy consumption in 2000",
        "generated_query": "What percentage of global energy came from renewables in 2000?"
    }}
]

Examples that should return an empty list []:
- "The information is not available in the provided context"
- "The provided context does not contain information about ..."
- "The context information does not provide details on ..."
- "1585" (just a number)
- "$35.78" (just a price)
- "March 15, 1985" (just a date)
- "2000-09-15" (just a date)
- "renewable energy" (just a concept)
- "the dark knight" (just a movie title)
- "John Smith, Mary Johnson, and David Chen" (just names)"""
            
            response = llm.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt + 1 < max_attempts:
                time.sleep(1)
                print(f"Retry {attempt + 1}: {str(e)}")
            else:
                print(f"Failed after {max_attempts} attempts: {str(e)}")
                return []

def verify_detail(detail: str, generated_query: str, item_id: int) -> tuple[str, str]:
    db = chromadb.PersistentClient(path="../generate_contexts/chromadb")
    
    collection_params = {
        "name": f"collection_{item_id}",
        "embedding_function": chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="../generate_contexts/model/dunzhang/stella_en_1.5B_v5",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    }
    
    chroma_collection = db.get_collection(**collection_params)
    
    QueryResults = chroma_collection.query(
        query_texts=[generated_query],
        n_results=6,
        include=["documents"]
    )
    
    contexts: Sequence[str] = QueryResults['documents'][0] if QueryResults['documents'][0] else [""]
    
    system_prompt = f"""Context information is below.
---------------------
{contexts}
---------------------"""
    
    user_prompt = f"""Given the context information and not prior knowledge, answer the query.
Query: {generated_query}
Answer: """
    
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer_to_generated_query = response.choices[0].message.content

    # Generate new_detail by comparing original detail with verification answer
    new_detail_prompt = f"""Original detail: {detail}
Verification query: {generated_query}
Verification answer: {answer_to_generated_query}

Based on the verification answer, please provide a corrected version of the original detail. 
If the original detail is accurate, return it unchanged. If it needs correction, provide the corrected version while maintaining a similar structure.

Corrected detail: """

    new_detail_response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": new_detail_prompt}]
    )
    
    return answer_to_generated_query, new_detail_response.choices[0].message.content

def generate_new_answer(original_answer: str, verified_details: list) -> str:
    prompt = f"""Original answer: {original_answer}

I have verified each detail in the original answer. For each detail, I have:
1. Extracted the detail from the original answer ("detail" key)
2. Generated a specific query about that detail ("generated_query" key)
3. Found a new answer to that query using reliable sources ("answer_to_generated_query" key)
4. The "new_detail" key contains the corrected version of the detail based on verification

Here are all the verified details:
{json.dumps(verified_details, indent=2)}

Please generate a new answer that:
1. Maintains the same style and tone as the original answer
2. Incorporates the verified information from "answer_to_generated_query"
3. Uses the "new_detail" when there are conflicts with the original detail
4. Remains factual and accurate based on the verification results

New answer: """
    
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def process_test_cases(use_pre_retrieval: bool = False, use_cot: bool = False):
    output_count = 0

    if use_pre_retrieval:
        input_dir = "pre-retrieval+cot" if use_cot else "pre-retrieval"
    else:
        input_dir = "cot" if use_cot else "none"

    if input_dir == "none":
        output_dir = "rig"
    else:
        output_dir = f"{input_dir}+rig"

    input_path = f"../results/{input_dir}/test_cases.jsonl"
    output_path = f"../results/{output_dir}/test_cases.jsonl"

    with open(input_path, "r") as input_file, \
        open(output_path, "w") as output_file:
        
        test_cases = [json.loads(line) for line in input_file]

        for item in test_cases:    
            # Step 1: Generate queries for details
            detail_queries = generate_detail_queries(item["answer"])
            
            if not detail_queries:
                print(f"Skipping verification for test case {item['id']} as no details are extracted")
                result_item = {
                    "id": item["id"],
                    "query": item["query"],
                    "ground_truth": item["ground_truth"],
                    "answer": item["answer"],
                    "details_verification": None,
                    "new_answer": item["answer"]
                }
            else:
                # Step 2: Verify each detail
                details_verification = []
                for dq in detail_queries:
                    answer_to_generated_query, new_detail = verify_detail(
                        dq["detail"],
                        dq["generated_query"],
                        item["id"]
                    )
                    verified_detail = {
                        "detail": dq["detail"],
                        "generated_query": dq["generated_query"],
                        "answer_to_generated_query": answer_to_generated_query,
                        "new_detail": new_detail
                    }
                    details_verification.append(verified_detail)
                
                # Step 3: Generate new answer
                new_answer = generate_new_answer(item["answer"], details_verification)
                
                # Create result object with verification results
                result_item = {
                    "id": item["id"],
                    "query": item["query"],
                    "ground_truth": item["ground_truth"],
                    "answer": item["answer"],
                    "details_verification": details_verification,
                    "new_answer": new_answer
                }

            output_file.write(json.dumps(result_item) + "\n")
            output_file.flush()
            
            output_count += 1
            print(f"Generated answers with RIG for {output_count} test cases")

    print(f"Answers generation with RIG completed. Results stored in {output_path}")

use_pre_retrieval = input("Used pre-retrieval for contexts generation? (T/F): ").upper() == 'T'
use_cot = input("Used Chain of Thought reasoning for answers generation? (T/F): ").upper() == 'T'
process_test_cases(use_pre_retrieval, use_cot)