#In this file, I defined some python functions that are used in the Agent.
#The extract_code function is used to extract python code from the answer of  the first_layer agent.
#The execute_code function is used to execute the code and get the results.
#The parse_response function is used to parse the response of the second_layer agent to a dict, 
#and then use the format_response function to print a clearly structured and user-friendly format for the user to read.


import pandas as pd


def extract_code(answer):
    # Check if the answer contains a valid Python code block
    if "```python" not in answer:
        print("Error: The answer does not contain a valid Python code block.") 
        return None 
        
    # Extract code block
    code_start = answer.find("```python") + 9
    code_end = answer.find("```", code_start)
    code = answer[code_start:code_end].strip()
    return code


def execute_code(answer, df):
    # Extract code from the answer
    code = extract_code(answer)
    
    try:
        # Create local namespace and execute code
        local_dict = {'df': df, 'pd': pd}
        code_reset = f"result = {code}.reset_index()"
        exec(code_reset, None, local_dict)
        return local_dict.get('result')
       
    except AttributeError as attr_error:
        # Handle attribute errors by executing the original code(the code that is not reset the index)
        code = f"result = {code}"
        exec(code, None, local_dict)
        return local_dict.get('result')
    
    except Exception as code_error:
        # Catch other exceptions,print the error message and return None
        print(f"Code execution error: {str(code_error)}")
        return None 
   

def parse_response(response):
    # Clean up the response by removing unnecessary characters
    response = response.replace('**', '')
    parts = response.split('\n\n')
    response_dict = {}

    # Split the response into parts and parse each part into a key-value pair
    for part in parts:
        idx = part.find(':')
        if idx != -1:
            key = part[:idx + 1].strip()  
            value = part[idx + 1:].strip()  
            response_dict[key] = value  

    return response_dict
  

def format_response(response):
    # Parse the response to get the dictionary
    response_dict = parse_response(response)
    print(f"📝 The question: {response_dict['The question:']}")
    print("=" * 50)
    
    # Split and format the Python code and execution result.
    python_result = response_dict['The python code and execution result:']
    code_start = python_result.find("'code':") + len("'code':")
    result_start = python_result.find("'result':") + len("'result':")
    
    code_content = python_result[code_start:result_start - len("'result':")].strip()
    result_content = python_result[result_start:].strip()
    
    # Print the python code and execution result in a clearly structured format
    print("\n💻 The python code:")
    print(code_content.replace('\\n', '\n').replace(',', ''))
    
    print("\n📊 The execution result:")
    print(result_content.replace('}', ''))
    
    print("\n✅ The concluding response:")
    print(response_dict['The concluding response:'])
    print("=" * 50 + "\n")
