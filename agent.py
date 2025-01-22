"""_summary_

This file defines the CodeRAGAgent and InterpRAGAgent classes, which are responsible for processing queries
and generating responses using a retrieval-augmented generation approach.

Functionality:
- The CodeRAGAgent class retrieves relevant context based on a query, formats a prompt, invokes a model,
  and executes the generated code to obtain results.
- The InterpRAGAgent class processes context(the results of the CodeRAGAgent) and queries to generate responses using a model,
  and formats the output for user-friendly presentation.

Usage:
For the CodeRAGAgent:
1. Create an instance of the CodeRAGAgent class with the necessary parameters:
   ```python
   agent = CodeRAGAgent(retriever, prompt, model, df)
   ```
2. Call the invoke method with a query to get the generated code and execution result:
   ```python
   result = agent.invoke(query)
   ```

For the InterpRAGAgent:
1. Create an instance similarly:
   ```python
   interp_agent = InterpRAGAgent(prompt, model)
   ```
2. Call the invoke method with context(the results of the CodeRAGAgent) and query to get the formatted response:
   ```python
   response = interp_agent.invoke(context, query)
   ```
"""


from execute import execute_code, format_response


class CodeRAGAgent:
    def __init__(self, retriever, prompt, model, df):
        # Initializes the CodeRAGAgent with a retriever, prompt template, model, and DataFrame.
        self.retriever = retriever
        self.prompt = prompt
        self.model = model
        self.df = df

    def processor(self, query):
        # Complement the RAG workflow
        context = self.retriever.retrieve_schema(query, self.df)
        prompt_output = self.prompt.format(context=context, question=query)
        model_output = self.model.invoke(prompt_output)
        return model_output
    
    def invoke(self, query):
        # Execute the code and get the result
        result = None
        max_attempts = 3  
        attempts = 0
        
        # Passes the code from the processor method to execute_code function to obtain the result of the code, 
        # here I used a while loop to enhance the robustness of the agent
        while result is None and attempts < max_attempts:
            code = self.processor(query)
            result = execute_code(code, self.df)
            attempts += 1  
        
        return {'code': code, 'result': result}

    
class InterpRAGAgent:
    def __init__(self, prompt, model):
        # Initializes the InterpRAGAgent with a prompt template and model.
        self.prompt = prompt
        self.model = model

    def processor(self, context, query):
        # Complement the RAG workflow
        prompt_output = self.prompt.format(context=context, question=query)
        return self.model.invoke(prompt_output)
    
    def invoke(self, context, query):
        # complement the processor method and format the response.
        response = self.processor(context, query)
        return format_response(response)
        
  