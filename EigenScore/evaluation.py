import numpy as np
import re
import json
import unicodedata
from collections import Counter
from nltk.stem import PorterStemmer
from word2number import w2n
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def words_to_numbers(text: str) -> str:
    """Convert written numbers (e.g., 'twenty-five') to numerical form ('25')."""
    try:
        return str(w2n.word_to_num(text))
    except ValueError:
        return text  

def normalize_text(text: str) -> str:
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(words_to_numbers(word) for word in text.split())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(stemmer.stem(str(word)) for word in words)

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

def macro_f1_score(predictions: list[str], ground_truths: list[list[str]]) -> float:
    f1_scores = []

    for pred, gt_list in zip(predictions, ground_truths):
        max_f1 = max(f1_score(pred, gt) for gt in gt_list)
        f1_scores.append(max_f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0

def classify_answers(predictions: list[str], ground_truths: list[list[str]], threshold: float) -> list[bool]:
    """
    Classify each prediction as correct or incorrect based on the F1 score.
    """
    is_correct = []
    f1=[]
    for pred, gt_list in zip(predictions, ground_truths):
        max_f1 = max(f1_score(pred, gt) for gt in gt_list)
        f1.append(max_f1)
        
        is_correct.append(max_f1 >= threshold)

    
    return is_correct

def evaluate_threshold_from_json(file_path: str, threshold: float) -> float:
    """
    Load a JSON file and compute the accuracy at a given threshold.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    predictions = [entry["most_likely_answer"] for entry in data]
    ground_truths = [[entry["answerGT"]] + entry.get("additional_answers", []) for entry in data]
    
    is_correct = classify_answers(predictions, ground_truths, threshold)
    return is_correct

def evaluate_f1_from_json(file_path: str) -> float:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    predictions = [entry["most_likely_answer"] for entry in data]
    ground_truths = [[entry["answerGT"]] + entry.get("additional_answers", []) for entry in data]
    return macro_f1_score(predictions, ground_truths)

def find_best_threshold_detection(file_path, score_key):
    """
    Find the threshold that maximizes the accuracy.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file) 
    scores=[]
    for i in range(len(data)):
        scores.append(data[i][score_key])
    labels=evaluate_threshold_from_json(file_path, 0.5)
    
    thresholds = np.linspace(np.min(scores), np.max(scores), num=100)

    accuracies = []
    for threshold in thresholds:
        is_correct = scores<=threshold
        accuracy = np.mean(is_correct == labels)
        accuracies.append(accuracy)
    best_threshold = thresholds[np.argmax(accuracies)]
    best_score = accuracies[np.argmax(accuracies)]
    accuracies = []
    return best_threshold, best_score

def find_best_threshold_rag(rag, no_rag, score_key):
    """
    Find the threshold that maximizes the accuracy.
    """
    with open(rag, 'r', encoding='utf-8') as file:
        data_rag = json.load(file)
    with open(no_rag, 'r', encoding='utf-8') as file:
        data_no_rag = json.load(file)
    
    scores_no_rag=[entry[score_key] for entry in data_no_rag]
    if score_key=="lexical_similarity":
        scores_no_rag = -np.array(scores_no_rag)
    f1 = []
    n = []
    thresholds = np.linspace(np.min(scores_no_rag), np.max(scores_no_rag), num=100)
    for threshold in thresholds:
        predictions = []
        n_rag = 0
        for i in range(len(data_no_rag)):
            if data_no_rag[i][score_key] <= threshold:
                predictions.append(data_no_rag[i]["most_likely_answer"])
            else:
                predictions.append(data_rag[i]["most_likely_answer"])
                n_rag += 1

        f1.append(macro_f1_score(predictions, [[entry["answerGT"]] + entry.get("additional_answers", []) for entry in data_no_rag]))
        n.append(n_rag/len(data_no_rag))

    best_f1 = f1[np.argmax(f1)]
    best_threshold = thresholds[np.argmax(f1)]
    best_n = n[np.argmax(f1)]
    return best_f1, best_threshold, best_n, f1, thresholds, n

def evaluate_f1_from_hybrid(rag, no_rag, score_key):
    """
    Load a JSON file and compute the macro-averaged F1 score.
    """
    with open(rag, 'r', encoding='utf-8') as file:
        data_rag = json.load(file)
    with open(no_rag, 'r', encoding='utf-8') as file:
        data_no_rag = json.load(file)
    
    #threshold, best_score = find_best_threshold_detection(no_rag, score_key)
    scores = [el[score_key] for el in data_no_rag]
    if score_key=="lexical_similarity":
        scores = -np.array(scores)
    labels = 1-np.array(evaluate_threshold_from_json(no_rag, 0.5))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    distances = np.sqrt(tpr*(1-fpr))
    optimal_idx = np.argmax(distances)
    threshold = thresholds[optimal_idx]
    predictions = [
    data_no_rag[i]["most_likely_answer"] if data_no_rag[i][score_key] < threshold
    else data_rag[i]["most_likely_answer"]
    for i in range(len(data_no_rag))
]

    ground_truths = [[entry["answerGT"]] + entry.get("additional_answers", []) for entry in data_no_rag]
    
    return macro_f1_score(predictions, ground_truths), threshold

