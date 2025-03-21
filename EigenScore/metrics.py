# Code adapted from https://github.com/alibaba/eigenscore/tree/main
# Original author: Alibaba

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import MinCovDet
from rouge_score import rouge_scorer
import heapq

rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def getRouge(rouge, generations, answers):
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure
    return RoughL

def get_perplexity_score(scores):
    perplexity = 0.0
    for logits in scores:
        conf = torch.max(logits.softmax(1)).cpu().item()
        perplexity += np.log(conf)
    perplexity = -1.0 * perplexity/len(scores)
    return perplexity


def get_energy_score(scores):
    avg_energy = 0.0
    for logits in scores:
        energy = - torch.logsumexp(logits[0], dim=0, keepdim=False).item()
        avg_energy += energy
    avg_energy = avg_energy/len(scores)
    return avg_energy

 
def get_entropy_score(batch_scores, num_tokens):  
    Conf = []
    for logits in batch_scores:
        conf, index = torch.max(logits.softmax(1), dim=1)
        Conf.append(conf.cpu().numpy())
    Conf = np.array(Conf)
    Conf = Conf + 1e-6
    entropy = -1.0 * np.sum(np.log(Conf))/logits.shape[0]
    return entropy


def get_lenghthNormalized_entropy(batch_scores, num_tokens):  
    seq_entropy = np.zeros(len(num_tokens))
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind]
    normalized_entropy = -1.0* normalized_entropy/len(num_tokens)
    return normalized_entropy


def getLexicalSim(generated_texts):
    LexicalSim = 0
    for i in range(len(generated_texts)):
        for j in range(len(generated_texts)):
            if j<=i:
                continue
            LexicalSim += getRouge(rougeEvaluator, generated_texts[i], generated_texts[j])
    LexicalSim = LexicalSim/(len(generated_texts)*(len(generated_texts)-1)/2)
    return LexicalSim


def getEigenIndicator_v0(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -1
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind,:] = hidden_states[num_tokens[ind]-2][selected_layer][ind,0,:]
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s




    






