from llm_config import *
from llama_index.core.node_parser import SentenceSplitter
from bs4 import BeautifulSoup
from typing import List, Dict
import chromadb
import bz2
import json
import chromadb.utils.embedding_functions
import torch

EMBED_MODEL="model/dunzhang/stella_en_1.5B_v5"
CHUNK_SIZE=3072
CHUNK_OVERLAP=int(0.2 * CHUNK_SIZE)
SIMILARITY_TOP_K=6
DISTANCE_METRIC="cosine"

def generate_queries(query: str, num_queries: int = 4) -> List[str]:
    system_prompt = "Generate multiple search queries based on the input query. Be specific and diverse."
    user_prompt = f"Generate {num_queries} different search queries related to: {query}"

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    queries = response.choices[0].message.content.strip().split("\n")
    return queries[:num_queries]

def reciprocal_rank_fusion(search_results_dict: Dict[str, Dict[str, float]], k: int = 60) -> Dict[str, float]:
    fused_scores = {}

    for _, doc_scores in search_results_dict.items():
        for rank, (doc, _) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + k)
    
    return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

def generate_contexts(embed_model, chunk_size, chunk_overlap, similarity_top_k, distance_metric, use_pre_retrieval: bool = False):
    db = chromadb.PersistentClient(path="chromadb")

    with open("random_nums.txt", "r") as f:
        random_nums = set(map(int, f.readlines()[:300]))

    generated = 0
    
    output_path = "../results/contexts_with_pre-retrieval.jsonl" if use_pre_retrieval else "../results/contexts_without_pre-retrieval.jsonl"

    with bz2.open("crag_task_3_dev_v4.jsonl.bz2", "r") as input_file, \
         open(output_path, "w") as output_file:
        
        for line_number, line in enumerate(input_file):
            if line_number not in random_nums:
                continue

            item = json.loads(line)

            collection_params = {
                "name": f"collection_{line_number}",
                "embedding_function": chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embed_model,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            }

            try:
                chroma_collection = db.get_collection(**collection_params)
            except Exception:
                if distance_metric != "l2":
                    collection_params["metadata"] = {"hnsw:space": distance_metric}

                chroma_collection = db.create_collection(**collection_params)

                sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                documents = []
                metadatas = []
                ids = []
                for result in item['search_results']:
                    soup = BeautifulSoup(result['page_result'], 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    
                    chunks = sentence_splitter.split_text(text)
                    
                    for chunk_id, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "page_name": str(result['page_name']),
                            "page_url": str(result['page_url']),
                            "page_last_modified": str(result['page_last_modified']),
                            "interaction_id": str(item['interaction_id'])
                        })
                        ids.append(f"{str(result['page_url'])}_{str(item['interaction_id'])}_{chunk_id}")
                
                chroma_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            if use_pre_retrieval:
                # Generate multiple queries
                queries = generate_queries(item['query'])

                # Get search results for each query
                all_results = {}
                for query in queries:
                    query_results = chroma_collection.query(
                        query_texts=[query],
                        n_results=similarity_top_k,
                        include=["documents", "distances"]
                    )
                    
                    # Create score dictionary for this query's results
                    scores_dict = {}
                    for doc, distance in zip(query_results['documents'][0], query_results['distances'][0]):
                        scores_dict[doc] = 1 - distance  # Convert distance to similarity score
                    all_results[query] = scores_dict

                # Combine results using reciprocal rank fusion
                fused_results = reciprocal_rank_fusion(all_results)
                
                # Take top K contexts after fusion
                contexts = list(fused_results.keys())[:similarity_top_k] if fused_results else [""]
            else:
                query_results = chroma_collection.query(
                    query_texts=[item['query']],
                    n_results=similarity_top_k,
                    include=["documents"]
                )
                
                contexts = query_results['documents'][0] if query_results['documents'][0] else [""]
            
            contexts_item = {
                "id": line_number,
                "query": item['query'],
                "ground_truth": item['answer'],
                "contexts": contexts
            }

            output_file.write(json.dumps(contexts_item) + '\n')
            output_file.flush()

            generated += 1
            use_pre_retrieval_status = "with pre-retrieval" if use_pre_retrieval else "without pre-retrieval"
            print(f"Generated contexts {use_pre_retrieval_status} for {generated} test cases")

            random_nums.remove(line_number)
            if len(random_nums) == 0:
                break
    
    print(f"Contexts generation completed. Results stored in {output_path}")

use_pre_retrieval = input("Use pre-retrieval for contexts generation? (T/F): ").upper() == 'T'
generate_contexts(EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K, DISTANCE_METRIC, use_pre_retrieval)