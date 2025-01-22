"""_summary_

This file serves as the entry point for the application, orchestrating the main functionality
of the retrieval-augmented generation system. It initializes necessary components(uses the best 
parameter combination chosen by the test_evaluate.ipynb file), processes input data,
and handles user queries to generate responses.

Functionality:
- The main function reads a DataFrame from a CSV file, initializes the retriever, prompt 
  and model(instances of the classes/functions defined in the retriever.py, prompt.py, 
  and model.py files respectively),and sets up the agents for code generation and interpretation.
- It processes a predefined query to retrieve relevant information and generate a response.
- The application can be run directly, executing the main function.

Usage:
1. Ensure the required CSV file ('data/product.csv') is available in the working directory.
2. Run the script to execute the main function:
   ```bash
   python main.py
   ```
3. The application will process the query and output the results to the console.
4. You can call the main function with your desired DataFrame and query:
   ```python
   main(df=pd.read_csv('your_file.csv'), query="Your question here")
   ```
"""


import pandas as pd
from langchain_community.vectorstores import FAISS

from model import Model
from retriever import Retriever
from prompts import get_prompt, combined_template, interp_template
from agent import CodeRAGAgent, InterpRAGAgent


def main(df=pd.read_csv('data/product.csv'),
         query="What is the average price of the products?"):
    # Initialize components with the best parameter combination tested in the test_evaluate notebook
    retriever = Retriever(mode='hybrid', 
                          embed_model_name="BAAI/bge-small-en-v1.5", 
                          db=FAISS,
                          top_k=5)
    code_model = Model(model_name="llama-3.3-70b-versatile",
                       temperature=0.1,
                       top_p=0.1)
    prompt = get_prompt(combined_template)
    processor = CodeRAGAgent(retriever, prompt, code_model, df)
    interp_prompt = get_prompt(interp_template)
    #interp_model = Model(model_name="llama-3.3-70b-versatile",temperature=0.2,top_p=0.1)
    interp = InterpRAGAgent(interp_prompt, code_model)

    # Process the query through the CodeRAGAgent to generate code and execution result
    ctx = processor.invoke(query)
    # Process the generated code and execution result through the InterpRAGAgent to generate a response
    interp.invoke(ctx, query)
    return None


if __name__ == "__main__":
    main()