from llm_config import *
import json
import time

INSTRUCTIONS = """
# Task: 
You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False" unless the Ground Truth is "invalid question".
3. For cases where Ground Truth is "invalid question":
   - "Accuracy" should be "True" if the prediction either:
     a) States explicitly that it's an invalid question
     b) Explains why the premise of the question is incorrect
     c) States that the information is not accurate or doesn't exist
     d) Indicates that the context or available information doesn't support the premise
# Output: 
Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".
"""

IN_CONTEXT_EXAMPLES = """
# Examples:
Question: how many seconds is 3 minutes 15 seconds?
Ground truth: ["195 seconds"]
Prediction: 3 minutes 15 seconds is 195 seconds.
Accuracy: True

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
Accuracy: False

Question: Who played Sheldon in Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Prediction: I am sorry I don't know.
Accuracy: False

Question: How many gold medals did Michelle Kwan win in the Olympics?
Ground truth: "invalid question"
Prediction: Michelle Kwan has not won any Olympic gold medals.
Accuracy: True

Question: Which states have universal healthcare for all residents?
Ground truth: "invalid question"
Prediction: No U.S. state currently has a universal healthcare program for all residents.
Accuracy: True

Question: Which five states have successfully implemented universal healthcare?
Ground truth: "invalid question"
Prediction: The context information does not provide specific details about any U.S. states that have successfully implemented universal healthcare programs.
Accuracy: True
"""

def llm_as_judge_evaluate(stage):
    test_cases_path = f"../results/{stage}/test_cases.jsonl"
    judge_results_path = f"../results/{stage}/judge_results.jsonl"
    
    with open(test_cases_path, "r") as test_cases_file, \
         open(judge_results_path, "w") as judge_results_file:
        
        test_cases = [json.loads(line) for line in test_cases_file]

        for item in test_cases:
            test_case_id = item["id"]
            judge_result = {"id": test_case_id, "accuracy": False}
            query = item["query"]
            ground_truth = item["ground_truth"].strip()
            
            if "new_answer" in item:
                prediction = item["new_answer"]
            else:
                prediction = item["answer"]

            if prediction is None:
                print(f"Test case {test_case_id} evaluated, accuracy: {judge_result['accuracy']}")
            else:
                prediction = prediction.strip()

                ground_truth_lowercase = ground_truth.lower()
                prediction_lowercase = prediction.lower()

                messages = [
                    {"role": "system", "content": INSTRUCTIONS + IN_CONTEXT_EXAMPLES},
                    {"role": "user", "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"}
                ]
                
                if "i don't know" in prediction_lowercase:
                    pass
                elif prediction_lowercase == ground_truth_lowercase:
                    judge_result["accuracy"] = True
                else:
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            response = llm.chat.completions.create(
                                model=MODEL_NAME,
                                temperature=TEMPERATURE,
                                messages=messages,
                                response_format={"type": "json_object"}
                            )

                            response = response.choices[0].message.content
                            response_lower = response.lower()
                            model_resp = json.loads(response_lower)
                            
                            if "accuracy" in model_resp and (
                                (model_resp["accuracy"] is True)
                                or (
                                    isinstance(model_resp["accuracy"], str)
                                    and model_resp["accuracy"].lower() == "true"
                                )
                            ):
                                judge_result["accuracy"] = True
                            
                            result = judge_result["accuracy"]
                            print(f"Test case {test_case_id} evaluated, accuracy: {result}")
                            break
                        except Exception as e:
                            if attempt + 1 < max_attempts:
                                time.sleep(1)
                                print(f"Retry {attempt + 1}: {str(e)}")
                            else:
                                print(f"Failed to evaluate test case {test_case_id} after {max_attempts} attempts: {str(e)}")
            
            judge_results_file.write(json.dumps(judge_result) + "\n")
            judge_results_file.flush()
    
    print(f"LLM as judge evaluation completed. Results stored in {judge_results_path}")

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
llm_as_judge_evaluate(stage)