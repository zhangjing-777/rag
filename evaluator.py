from bert_score import score       



class RetrievalEvaluator:
    def __init__(self, retrieved_docs, relevant_docs):
        self.retrieved_docs = retrieved_docs
        self.relevant_docs = relevant_docs

    def recall_at_k(self):
        relevant_retrieved = len(set(self.retrieved_docs) & set(self.relevant_docs))
        return relevant_retrieved / len(self.relevant_docs)

    def mean_reciprocal_rank(self):
        for i, doc in enumerate(self.retrieved_docs, start=1):
            if doc in self.relevant_docs:
                return 1 / i
        return 0
    
    def evaluate(self):
        return {
            f"Recall@k": self.recall_at_k(),
            "MRR": self.mean_reciprocal_rank()
        }



class GenerCodeEvaluator:
    def __init__(self, generated, reference):
        self.generated = generated
        self.reference = reference

    def exact_match(self):
        return int(self.generated.strip().lower() == self.reference.strip().lower())
    
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
    
    def evaluate(self):
        return {
            "Exact Match": self.exact_match(),
            "F1 Score": self.f1_score()
        }
    


def bert_score_f1(generated, reference):
    P, R, F1 = score(generated, reference, lang="zh")
    return round(float(F1.mean().item()),2)

        