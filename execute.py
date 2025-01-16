import pandas as pd


def extract_code(answer):
    if "```python" not in answer:
        print("Error: The answer does not contain a valid Python code block.") 
        return None 
        
    # Extract code block
    code_start = answer.find("```python") + 9
    code_end = answer.find("```", code_start)
    code = answer[code_start:code_end].strip()
    return code


def execute_code(answer, df):
    code = extract_code(answer)
    
    try:
        # Create local namespace and execute code
        local_dict = {'df': df, 'pd': pd}
        code_reset = f"result = {code}.reset_index()"
        exec(code_reset, None, local_dict)
        return local_dict.get('result')
       
    except AttributeError as attr_error:
        code = f"result = {code}"
        exec(code, None, local_dict)
        return local_dict.get('result')
    
    except Exception as code_error:
        print(f"Code execution error: {str(code_error)}")
        return None 
   

def parse_response(response):
    response = response.replace('**', '')
    parts = response.split('\n\n')
    response_dict = {}

    for part in parts:
        idx = part.find(':')
        if idx != -1:
            key = part[:idx + 1].strip()  
            value = part[idx + 1:].strip()  
            response_dict[key] = value  

    return response_dict
  

def format_response(response):
    response_dict = parse_response(response)
    print(f"📝 The question: {response_dict['The question:']}")
    print("=" * 50)
    
    # Split and format the Python code and execution result.
    python_result = response_dict['The python code and execution result:']
    code_start = python_result.find("'code':") + len("'code':")
    result_start = python_result.find("'result':") + len("'result':")
    
    code_content = python_result[code_start:result_start - len("'result':")].strip()
    result_content = python_result[result_start:].strip()
    
    print("\n💻 The python code:")
    print(code_content.replace('\\n', '\n').replace(',', ''))
    
    print("\n📊 The execution result:")
    print(result_content.replace('}', ''))
    
    print("\n✅ The concluding response:")
    print(response_dict['The concluding response:'])
    print("=" * 50 + "\n")
