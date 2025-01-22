"""_summary_

This file defines a Model class for handling different types of language models, including local and remote models.

Functionality:
- The Model class supports calling language models through the OllamaLLM and ChatGroq libraries.
- The invoke method automatically selects either the local model or the remote model based on the format of the model name (whether it contains a '-').

Usage:
1. Import this file and create an instance of the Model class:
   ```python
   model_instance = Model(model_name="your_model_name", additional_param=value)
   ```
2. Call the invoke method and pass in a prompt:
   ```python
   response = model_instance.invoke("Your prompt content")
   ```
3. Depending on the model, the invoke method will return the corresponding result:
   - If the model name contains '-', the remote model (ChatGroq) will be used.
   - Otherwise, the local model (OllamaLLM) will be used.

Note: When using the remote model, a valid GROQ_API_KEY is required.
"""

from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

GROQ_API_KEY = "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"

class Model:
    # Initializes an instance of the Model class
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    # Handles the prompt using the local model
    def local_model(self, prompt):
        llm = OllamaLLM(model=self.model_name, **self.kwargs)
        return llm.invoke(prompt)
    
    # Handles the prompt using the remote model
    def remote_model(self, prompt):
        llm = ChatGroq(model=self.model_name, api_key=GROQ_API_KEY, **self.kwargs)            
        return llm.invoke(prompt).content
    
    # Selects to use either the local or remote model based on the model name
    def invoke(self, prompt):
        if '-' in self.model_name:
            return self.remote_model(prompt)
        else:
            return self.local_model(prompt)
    
    
