# Project Introduction:
This project focuses on building a tabular data Q&A system by using double rag with language models.

# Set up environment
pip install -r requirements.txt

# Modules Introduction:
- `main.py`: Main script to execute experiments.
- `agent.py`: Implementation of the double RAG agent -- CodeRAGAgent and InterpRAGAgent.
- `model.py`: Manages calls to LLM APIs from langchain_ollama and ChatGroq.
- `prompts.py`: Gets prompts for CodeRAGAgent and InterpRAGAgent from langchain.prompts.
- `retriever.py`: Manages calls to Embedding Models from langchain_huggingface, builds database and performs schema retrieval.
- `execute.py`: Executes the pandas code, and formats the responses.
- `evaluator.py`: Definds recall_at_k and MRR metrics for retrieval model, exact_match and f1_score metrics for code generate model, bert_score_f1 metric for response generate model.


- `test_func.ipynb`: Tests the implementation of every function in above .py files.
- `test_evaluate.ipynb`: Tests different LLMs, Embedding Models, DBs and Parameters using the metrics definded in the evaluator.py file.Choose the best combination.
- `test_case.ipynb`:  Tests cases using the best combination chosen by test_evaluate.py file.


# System Architecture Diagram
![System Architecture Diagram](architecture.001.jpeg)
