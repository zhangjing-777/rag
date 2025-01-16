import pandas as pd
from langchain_community.vectorstores import FAISS

from model import Model
from retriever import Retriever
from prompts import get_prompt, combined_template, interp_template
from agent import CodeRAGAgent, InterpRAGAgent


def main(df=pd.read_csv('sample_product.csv'),
         query="What is the average price of the products?"):

    retriever = Retriever(mode='hybrid', 
                          embed_model_name="BAAI/bge-small-en-v1.5", 
                          db=FAISS,
                          top_k=5)
    code_model = Model(model_name="llama-3.3-70b-versatile",
                       temperature=0.2,
                       top_p=0.1)
    prompt = get_prompt(combined_template)
    processor = CodeRAGAgent(retriever, prompt, code_model, df)
    interp_prompt = get_prompt(interp_template)
    #interp_model = Model(model_name="llama-3.3-70b-versatile",temperature=0.2,top_p=0.1)
    interp = InterpRAGAgent(interp_prompt, code_model)

    ctx = processor.invoke(query)
    interp.invoke(ctx, query)
    return None


if __name__ == "__main__":
    main()