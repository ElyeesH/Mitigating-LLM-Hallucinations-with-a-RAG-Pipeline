import re
import json
import datasets
from Models.llama import load_llama
from data.triviaqa import get_dataset

_, tokenizer = load_llama()
data = get_dataset(tokenizer)

def postprocess(filename):
    with open(filename+'.out', 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r"Question:\s*(.*?)\n.*?AnswerGT:\s*(.*?)\n.*?MostLikelyAns:\s*(.*?)\n.*?Perplexity:\s*([\d.e-]+)\n.*?Energy:\s*([-?\d.e-]+)\n.*?NormalizedEntropy:\s*([-?\d.e-]+)\n.*?LexicalSimilarity:\s*([-?\d.e-]+)\n.*?EigenScore:\s*([-?\d.e-]+)\n"

    matches = re.findall(pattern, content, re.DOTALL)
    qa_list = []
    for question, answerGT, most_likely_answer, perplexity, energy, ne, ls, es in matches:
        qa_list.append({
                "question": question.strip(),
                "answerGT": answerGT.strip(),
                "most_likely_answer": most_likely_answer.strip(),
                "perplexity": float(perplexity),  
                "energy": float(energy),
                "normalized_entropy": float(ne),
                "lexical_similarity": float(ls),
                "eigen_score": float(es)
            })

    for i in range(len(qa_list)):
        qa_list[i]['additional_answers'] = data[i]['additional_answers']
    with open(filename+'.json', 'w', encoding='utf-8') as outfile:
        json.dump(qa_list, outfile, indent=4, ensure_ascii=False)
    return qa_list





