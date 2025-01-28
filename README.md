# Integrated RAG System
A system of Retrieval Augmented Generation (RAG) integrated with Pre-retrieval, Chain-of-Thought (CoT), and Retrieval Interleaved Generation (RIG). Accuracies are evaluated on a dataset of CRAG benchmark by LLM as judge.

## System Components

### Retrieval Augmented Generation (RAG)
RAG combines retrieval of relevant documents with text generation to produce more accurate and factual responses. The system first retrieves relevant contexts from a knowledge base, then uses them to inform the generation of responses.

### Pre-retrieval
Pre-retrieval enhances the initial documents retrieval process by generating multiple search queries for a single user question. These queries are then combined using reciprocal rank fusion to obtain the most relevant contexts before generation.

### Chain-of-Thought (CoT)
Chain-of-Thought prompting enables the model to break down complex questions into smaller, manageable steps. By decomposing questions into follow-up questions and intermediate answers, the system can arrive at more accurate final answers through structured reasoning.

### Retrieval Interleaved Generation (RIG)
RIG is intended to improve accuracies by verifying and refining generated responses. It:
1. Extract key details from initial answers
2. Generate specific queries to verify each detail
3. Retrieve additional contexts for verification
4. Synthesize a new, more accurate answer based on verified information

## Dataset Description
The system uses the CRAG Task 3 dataset (crag_task_3_dev_v4.jsonl.bz2), which provides comprehensive data for RAG evaluation. Each entry in the dataset follows this JSON structure:
```json
{
    "interaction_id": "string",     // Unique identifier
    "query_time": "string",         // Query timestamp
    "domain": "string",             // Category: finance/music/movie/sports/open
    "question_type": "string",      // Type: simple/comparison/aggregation/etc
    "static_or_dynamic": "string",  // Data volatility: static/slow-changing/fast-changing/real-time
    "query": "string",              // The actual question
    "answer": "string",             // Gold standard answer
    "alt_ans": ["string"],          // Alternative valid answers
    "split": "integer",             // 0=validation, 1=public test
    "search_results": [{            // Up to 50 HTML pages per query
        "page_name": "string",
        "page_url": "string",
        "page_snippet": "string",   // Brief content summary
        "page_result": "string",    // Full HTML content
        "page_last_modified": "string"
    }]
}
```
The dataset includes diverse question types across multiple domains, with each query supported by up to 50 web pages for contexts retrieval

## Steps to Evaluate

### Step 1: Download dataset
crag_task_3_dev_v4.jsonl.bz2 (7.19GB)  
https://mycuhk-my.sharepoint.com/:u:/g/personal/1155174302_link_cuhk_edu_hk/ETGwLIbMLwBHuSuK2ZU3LV4BYwzTRYfA5JbsrdcRiGzwJA?e=xgYuQO  
Store the bz2 file in generate_contexts directory

### Step 2: Setup environment
Choose either option A (Docker) or option B (pip):

#### Option A: Docker
```bash
docker build -t integrated-rag-system .
docker run --gpus all -it -v .:/workspace integrated-rag-system
```
Restart container if it has been started for a while or when the process of generating contexts is stuck

#### Option B: pip
```bash
pip install llama-index==0.11.2 huggingface_hub[hf_transfer]==0.24.6 \
    sentence-transformers==3.0.1 torch==2.4.0 chromadb==0.5.5 bs4==0.0.2
```

### Step 3: Download local embedding model
```bash
huggingface-cli login
<Hugging Face Access Token>
cd generate_contexts
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    dunzhang/stella_en_1.5B_v5 \
    --local-dir model/dunzhang/stella_en_1.5B_v5
```

### Step 4: Generate contexts based on 300 randomly selected test cases from dataset
```bash
python3 generate_contexts.py
```
Set LLM configuration in llm_config.py before execution  
Choose whether to use pre-retrieval after execution

### Step 5: Generate answers
```bash
cd ../generate_answers
python3 generate_answers.py
```
Set LLM configuration in llm_config.py before execution  
Choose whether pre-retrieval was used and whether to use CoT after execution

### Step 6: Run RIG
```bash
python3 rig.py
```
Choose whether pre-retrieval and CoT were used after execution

### Step 7: Run LLM as judge on answers generated for evaluation
```bash
cd ../llm_as_judge
python3 llm_as_judge.py
```
Set LLM configuration in llm_config.py before execution  
Choose which stage of answers to evaluate after execution

## Evaluation Results
Average scores of LLM as judge (True as 1, False as 0)  
Corrected to 4 significant digits  
  
none: 0.47  
pre-retrieval: 0.54  
cot: 0.5267  
rig: 0.4767  
pre-retrieval+cot: 0.5567  
cot+rig: 0.5067  
pre-retrieval+rig: 0.5233  
pre-retrieval+cot+rig: 0.5433  
  
Therefore, pre-retrieval+cot is the best combination  
There is a probability to synthesize a less accurate answer with RIG  
Applications based on the evaluated system are being developed

## Citations
[1] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga, Anuj Kumar, Wen-tau Yih, Xin Luna Dong. 2024. CRAG -- Comprehensive RAG Benchmark. arXiv preprint arXiv:2406.04744.  

[2] AIcrowd | Meta KDD Cup 2024 - CRAG: Comprehensive RAG Benchmark - End-to-End Retrieval-Augmented Generation, 2024. https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/meta-kdd-cup-24-crag-end-to-end-retrieval-augmented-generation (accessed Oct. 20, 2024).