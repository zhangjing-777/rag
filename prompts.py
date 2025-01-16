from langchain.prompts import PromptTemplate


def get_prompt(template):                
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]  
    )
    return prompt
  
            
combined_template = """You are a pandas dataframe query code generator. The name of the dataframe is `df`. Your task is to answer the question with pandas dataframe operation code.

              The question is: {question}
              The context is: {context}
              
              **Thinking Process:**
              1. Identify the key information needed to answer the question.
              2. Determine which columns in the dataframe are relevant to the question.
              3. Consider the appropriate pandas operation code and the types of these columns to ensure the correct code is applied.
              4. Formulate the final code in ```python``` block.
               
              **Important Note & Examples:**
              Important Note1:You only have a context that includes the column names and dtypes of the columns. There is no row or cell information available. Therefore, when the user's question involves specific cell information, you should provide the code that corresponds to the entire column's data instead of specific values.
                              Specifically, avoid using operations like `df[df[column] == value]` since the exact values are unknown. Instead, use operations that summarize or count the entire column, such as `df['column_name'].value_counts()` or `df.groupby('column_name')['column_name'].mean()`. 
              Example1:
                 Question: "If students who completed preparation have a better writing score?"
                 Thinking:
                   1. I need to find the 'writing score' for completed preparation and the 'writing score' for not completed preparation separately.
                   2. I need to use 'writing score' column and 'preparation' column, but there is no 'preparation' column in the `df`, so I need to choose a similar column 'test preparation course' to alternative.
                   3. I haven't the values of 'test preparation course' column, so I will get all values with grouping by 'test preparation course' column and using mean() on 'writing score' column,and the type of 'writing score' is `int64`,the type of 'test preparation course' is `object`, that will be a valid operation.
                   4. My final code is: df.groupby('test preparation course')['writing score'].mean(), and I need to return the code in ```python``` block. 
                 Answer: 
                 ```python
                 df.groupby('test preparation course')['writing score'].mean()
                 ```
                 
              Important Note2:If the keywords of the user's question are not the same as the column names of `df`, please use similar semantic matching.
              Example2:
                 Question: "What is the average arithmetic score?"
                 Thinking:
                   1. I need to find the average value of the 'arithmetic score' column.
                   2. There is no 'arithmetic score' column in the `df`, but there is a 'math score' column that is similar, so I can use the 'math score' column to calculate the average value.
                   3. I can use the `mean()` function, and the type of 'math score' is `int64`, that will be a valid operation for the column.
                   4. My final code is: df['math score'].mean(), and I need to return the code in ```python``` block.
                 Answer: 
                 ```python
                 df['math score'].mean()
                 ```
                 
              Important Note3:If the question refers to a summary or aggregate of multiple columns (like total/comprehensive scores), infer the appropriate calculation based on the context.
              Example3:
                 Question: "What's the best comprehensive score?"
                 Thinking:
                   1. I need to find the maximum value of comprehensive score.
                   2. There isn't 'comprehensive score' column in the `df`, but 'comprehensive score' can be the sum of 'reading score', 'writing score', and 'math score'.
                   3. I can use the `max()` and `sum()`function, and 'reading score' 'writing score' 'math score' are all of type `int64`,that will be a valid operation.
                   4. My final code is: df[['reading score', 'writing score', 'math score']].sum(axis=1).max(), and I need to return the code in ```python``` block.
                 Answer: 
                 ```python
                 df[['reading score', 'writing score', 'math score']].sum(axis=1).max()
                 ```
                 
              Important Note4:The purpose of generating code is to produce data that can answer the user's question. If it is not possible to give a code for data based on the question directly, then giving a code that can obtain detailed information will suffice.
              Example4:
                 Question: "What're the features of the student who has the best reading score?"
                 Thinking:
                   1. I need to find the features of the student who has the maximum reading score.
                   2. I can use 'features' column and 'reading score' column, but there isn't 'features' column in the `df`, and `features` can be the combination of all columns.
                   2. I can use the `idxmax()` function and `iloc` function to get the row of the student who has the maximum reading score, and the type of 'reading score' is `int64`, that will be a valid operation.
                   3. My final code is: df.iloc[df['reading score'].idxmax()], and I need to return the code in ```python``` block.
                 Answer: 
                 ```python
                 df.iloc[df['reading score'].idxmax()]
                 ```
                 
              Rules:
                1. Consider the column types when writing code.For example,you cannot use the `cor()` function between `object` type columns and `int` type columns.
                2. Ensure that the generated code is syntactically correct and complete, including all necessary parentheses and syntax.
                3. Just return complete executable code in ```python``` block using the existing `df`, and don't include any explanations or additional text.
       
              Now, please follow this thinking process to generate the code for the following question:
              Question: {question}

              **Answer:**
              ```python
              [write your pandas operation code here]
              ```
              """

interp_template ="""
      You are an intelligent data analysis assistant. 
      Your task is to extract the most relevant data from the user's question, the corresponding Python code, and its execution result, and provide a clear and concise one-sentence answer to the user.  
    
      The question: {question}
      The python code and execution result: {context}

      Please provide a clear and logical response to the user, strictly including only the following 3 sections and no others:
            - The question: {question}\n
            - The python code and execution result: {context}\n
            - The concluding response: a concluding response that is reasonable and reliable based on the user's question, the python code and execution result.\n

      Ensure that the output strictly adheres to this format without any additional text or sections.  
      example:
          The question: Which racial group has the best writing score?\n
          The python code and execution result: ['code': "```python\ndf.loc[df['writing score'].idxmax()]['race/ethnicity']\n```",
                                                'result': 'group D']\n
          The concluding response: The relative result indicates that Group D has achieved the highest writing scores, making it the racial group with the best writing performance.\n
"""
      
