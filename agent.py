from execute import execute_code, format_response


class CodeRAGAgent:
    def __init__(self, retriever, prompt, model, df):
        self.retriever = retriever
        self.prompt = prompt
        self.model = model
        self.df = df

    def processor(self, query):
        context = self.retriever.retrieve_schema(query, self.df)
        prompt_output = self.prompt.format(context=context, question=query)
        model_output = self.model.invoke(prompt_output)
        return model_output
    
    def invoke(self, query):
        result = None
        max_attempts = 3  
        attempts = 0
        
        while result is None and attempts < max_attempts:
            code = self.processor(query)
            result = execute_code(code, self.df)
            attempts += 1  
        
        return {'code': code, 'result': result}

    
class InterpRAGAgent:
    def __init__(self, prompt, model):
        self.prompt = prompt
        self.model = model

    def processor(self, context, query):
        prompt_output = self.prompt.format(context=context, question=query)
        return self.model.invoke(prompt_output)
    
    def invoke(self, context, query):
        response = self.processor(context, query)
        return format_response(response)
        
  