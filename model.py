from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM



GROQ_API_KEY = "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"

class Model:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        
    def local_model(self, prompt):
        llm = OllamaLLM(model=self.model_name, **self.kwargs)
        return llm.invoke(prompt)
    
    def remote_model(self, prompt):
        llm = ChatGroq(model=self.model_name, api_key=GROQ_API_KEY, **self.kwargs)            
        return llm.invoke(prompt).content
    
    def invoke(self, prompt):
        if '-' in self.model_name:
            return self.remote_model(prompt)
        else:
            return self.local_model(prompt)
    
    
