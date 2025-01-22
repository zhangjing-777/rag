"""_summary_

This file defines classes and functions for evaluating retrieval and generation models performance.
It includes metrics such as Recall@k and Mean Reciprocal Rank (MRR) for retrieval evaluation,
Exact Match and F1 Score for code generation evaluation,
and a function for computing the F1 score using the BERTScore method for textual generation evaluation.

Functionality:
- The RetrievalEvaluator class computes Recall@k and MRR metrics to assess the effectiveness of document retrieval systems.
- The GenerCodeEvaluator class calculates Exact Match and F1 Score metrics to evaluate the quality of generated code against reference code.
- The bert_score_f1 function computes the F1 score using the BERTScore method for comparing generated and reference texts.

Usage:
For retrieval evaluation:
1. Create an instance of the RetrievalEvaluator class with retrieved and relevant documents:
   ```python
   evaluator = RetrievalEvaluator(retrieved_docs, relevant_docs)
   ```
2. Call the evaluate method to get the evaluation metrics:
   ```python
   metrics = evaluator.evaluate()
   ```

For code generation evaluation:
1. Create an instance of the GenerCodeEvaluator class with generated code and reference code:
   ```python
   code_evaluator = GenerCodeEvaluator(generated_code, reference_code)
   ```
2. Call the evaluate method to get the Exact Match and F1 Score:
   ```python
   code_metrics = code_evaluator.evaluate()
   ```

For textual generation evaluation:
Use the bert_score_f1 function to compute the F1 score between generated text and reference text:
   ```python
   f1_score = bert_score_f1(generated_text, reference_text)
   ```
"""

from bert_score import score       


class RetrievalEvaluator:
    # Initializes the evaluator with retrieved and relevant documents.
    def __init__(self, retrieved_docs, relevant_docs):
        self.retrieved_docs = retrieved_docs
        self.relevant_docs = relevant_docs

    # Calculates Recall@k metric, which measures the proportion of relevant documents retrieved.
    def recall_at_k(self):
        relevant_retrieved = len(set(self.retrieved_docs) & set(self.relevant_docs))
        return relevant_retrieved / len(self.relevant_docs)

    # Calculates Mean Reciprocal Rank (MRR), which evaluates the rank of the first relevant document.
    def mean_reciprocal_rank(self):
        for i, doc in enumerate(self.retrieved_docs, start=1):
            if doc in self.relevant_docs:
                return 1 / i
        return 0

    # Returns a dictionary containing the evaluation metrics: Recall@k and MRR.
    def evaluate(self):
        return {
            f"Recall@k": self.recall_at_k(),
            "MRR": self.mean_reciprocal_rank()
        }



class GenerCodeEvaluator:
    # Initializes the evaluator with generated and reference code.
    def __init__(self, generated, reference):
        self.generated = generated
        self.reference = reference

    # Calculates Exact Match metric, which checks if the generated code matches the reference code exactly.
    def exact_match(self):
        return int(self.generated.strip().lower() == self.reference.strip().lower())
    
    # Calculates F1 Score, which is the harmonic mean of precision and recall for the generated code.
    def f1_score(self):
        gen_tokens = set(self.generated.split())
        ref_tokens = set(self.reference.split())
        common = gen_tokens & ref_tokens
        if not common:
            return 0
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1,2)
    
    # Returns a dictionary containing the evaluation metrics: Exact Match and F1 Score.
    def evaluate(self):
        return {
            "Exact Match": self.exact_match(),
            "F1 Score": self.f1_score()
        }
    


# Computes the F1 score using the BERTScore method for comparing generated and reference texts.
def bert_score_f1(generated, reference):
    # Set lang to "en" for English
    P, R, F1 = score(generated, reference, lang="en")
    return round(float(F1.mean().item()),2)

        