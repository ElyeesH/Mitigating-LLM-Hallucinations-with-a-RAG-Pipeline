
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random
import gc
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LEDForConditionalGeneration, LEDTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import sys
from pathlib import Path
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import numpy as np
import pandas as pd
import evaluate

from huggingface_hub import login

# change your base path
base_path = Path("/home/ensta/ensta-zguerguer/Mitigating-LLM-Hallucinations-with-a-RAG-Pipeline")

# Fix if you want to use a different path
output_path = base_path / "output"
data_path = base_path / "token_probability_approach"
output_path.mkdir(exist_ok=True)

"""## These are the libraries needed to run this notebook

## Deep Learning Installations
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install transformers datasets evaluate rouge_score
# %pip install --upgrade huggingface_hub
# %pip install accelerate -U
# %pip install transformers[torch]
# %pip install sentencepiece
# %pip install google
# %pip install protobuf




"""## Libraries"""


"""# Reading Dataset

`The method loadDataset receieves the path where the datasets json files of the HaluEval repository are. You just need to pass your path and the name of the dataset you are going to use.`

## Dataset Names:
- summarization
- dialogue
- qa
- general

"""

# As a recomendation keep these two with the same naming if you do not
# want to change many things
datasetName = "qa"
task = datasetName


def loadDataset(path="./data", datasetName="qa"):
  data = pd.read_json(
      data_path / (path + "/" + datasetName + "_data.json"), lines=True
  )
  return data


"""## For this particular example we are loading the qa_data.json since is the one that takes the less time to process in case you want to test quickly how it works."""

data = loadDataset(datasetName=datasetName)

data.head()

len(data)

"""# Setting Device to use the GPU

We use the T4 GPU in Colab since the heaviest computation for us is the inference of the LLM-Evaluator. Therefore, T4 seem as the better fit.
"""


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""## Generic LLMModel class to reuse the functionality of extracting the features.

"""


class LLMModel:
  def __init__(self):
    self.model = self.model.to(device)
    pass

  def getName(self) -> str:
    return self.model_name

  def getSanitizedName(self) -> str:
    return self.model_name.replace("/", "__")

  def generate(self, inpt):
    pass

  # Move in future commits this method to an utils.py
  def truncate_string_by_len(self, s, truncate_len):
    words = s.split()
    truncated_words = words[:-truncate_len] if truncate_len > 0 else words
    return " ".join(truncated_words)

  # Method to get the vocabulary probabilities of the LLM for a given token
  # on the generated text from LLM-Generator
  def getVocabProbsAtPos(self, pos, token_probs):
    sorted_probs, sorted_indices = torch.sort(
        token_probs[pos, :], descending=True)
    return sorted_probs

  def getMaxLength(self):
    return self.model.config.max_position_embeddings

  # By default knowledge is the empty string. If you want to add extra
  # knowledge you can do it like in the cases of the qa_data.json and
  # dialogue_data.json
  def extractFeatures(
      self,
      knowledge="",
      conditionted_text="",
      generated_text="",
      features_to_extract={},
  ):
    self.model.eval()

    total_len = len(knowledge) + len(conditionted_text) + len(generated_text)
    truncate_len = min(total_len - self.tokenizer.model_max_length, 0)

    # Truncate knowledge in case is too large
    knowledge = self.truncate_string_by_len(knowledge, truncate_len // 2)
    # Truncate text_A in case is too large
    conditionted_text = self.truncate_string_by_len(
        conditionted_text, truncate_len - (truncate_len // 2)
    )

    inputs = self.tokenizer(
        [knowledge + conditionted_text + generated_text],
        return_tensors="pt",
        max_length=self.getMaxLength(),
        truncation=True,
    )

    for key in inputs:
      inputs[key] = inputs[key].to(device)

    with torch.no_grad():
      outputs = self.model(**inputs)
      logits = outputs.logits

    probs = F.softmax(logits, dim=-1)
    probs = probs.to(device)

    tokens_generated_length = len(self.tokenizer.tokenize(generated_text))
    start_index = logits.shape[1] - tokens_generated_length
    conditional_probs = probs[0, start_index:]

    token_ids_generated = inputs["input_ids"][0, start_index:].tolist()
    token_probs_generated = [
        conditional_probs[i, tid].item()
        for i, tid in enumerate(token_ids_generated)
    ]

    tokens_generated = self.tokenizer.convert_ids_to_tokens(
        token_ids_generated)

    minimum_token_prob = min(token_probs_generated)
    average_token_prob = sum(
        token_probs_generated) / len(token_probs_generated)

    maximum_diff_with_vocab = -1
    minimum_vocab_extreme_diff = 100000000000

    if features_to_extract["MDVTP"] or features_to_extract["MMDVP"]:
      size = len(token_probs_generated)
      for pos in range(size):
        vocabProbs = self.getVocabProbsAtPos(pos, conditional_probs)
        maximum_diff_with_vocab = max(
            [
                maximum_diff_with_vocab,
                self.getDiffVocab(vocabProbs, token_probs_generated[pos]),
            ]
        )
        minimum_vocab_extreme_diff = min(
            [
                minimum_vocab_extreme_diff,
                self.getDiffMaximumWithMinimum(vocabProbs),
            ]
        )

    # allFeatures = [minimum_token_prob, average_token_prob, maximum_diff_with_vocab, minimum_vocab_extreme_diff]

    allFeatures = {
        "mtp": minimum_token_prob,
        "avgtp": average_token_prob,
        "MDVTP": maximum_diff_with_vocab,
        "MMDVP": minimum_vocab_extreme_diff,
    }

    selectedFeatures = {}
    for key, feature in features_to_extract.items():
      if feature:
        selectedFeatures[key] = allFeatures[key]

    return selectedFeatures

  def getDiffVocab(self, vocabProbs, tprob):
    return (vocabProbs[0] - tprob).item()

  def getDiffMaximumWithMinimum(self, vocabProbs):
    return (vocabProbs[0] - vocabProbs[-1]).item()


"""## Definition of the specific Models"""


class Gemma(LLMModel):
  def __init__(self):
    self.model_name = "google/gemma-7b-it"
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name
    )
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer(
        [inpt],
        max_length=self.getMaxLength(),
        return_tensors="pt",
        truncation=True)
    summary_ids = self.model.generate(inputs["input_ids"])

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


class LLama(LLMModel):
  def __init__(self):
    self.model_name = "meta-llama/Llama-2-7b-chat-hf"
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
    )
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer(
        [inpt], max_length=1024, return_tensors="pt", truncation=True
    )
    summary_ids = self.model.generate(inputs["input_ids"])

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


class Opt(LLMModel):
  def __init__(self):
    self.model_name = "facebook/opt-6.7b"
    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer(
        [inpt],
        max_length=self.getMaxLength(),
        return_tensors="pt",
        truncation=True)
    summary_ids = self.model.generate(inputs["input_ids"])

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


class Gptj(LLMModel):
  def __init__(self):
    self.model_name = "EleutherAI/gpt-j-6B"
    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer(
        [inpt],
        max_length=self.getMaxLength(),
        return_tensors="pt",
        truncation=True)
    summary_ids = self.model.generate(inputs["input_ids"])

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


class BartCNN(LLMModel):
  def __init__(self):
    self.model_name = "facebook/bart-large-cnn"
    self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
    self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer(
        [inpt],
        max_length=self.getMaxLength(),
        return_tensors="pt",
        truncation=True)
    summary_ids = self.model.generate(inputs["input_ids"])

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


class GPT2Generator(LLMModel):
  def __init__(self):
    self.model_name = "gpt2-large"
    self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
    self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer.encode(
        inpt,
        return_tensors="pt",
        max_length=self.getMaxLength(),
        truncation=True)
    output_ids = self.model.generate(
        inputs, max_length=1024, num_return_sequences=1
    )
    output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output


class LED(LLMModel):
  def __init__(self):
    self.model_name = "allenai/led-large-16384-arxiv"
    self.model = LEDForConditionalGeneration.from_pretrained(self.model_name)
    self.tokenizer = LEDTokenizer.from_pretrained(self.model_name)
    super().__init__()

  def generate(self, inpt):
    inputs = self.tokenizer(
        [inpt],
        max_length=self.getMaxLength(),
        return_tensors="pt",
        truncation=True)
    summary_ids = self.model.generate(inputs["input_ids"])

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


"""# The Dictionary `features_to_extract` defines which features will be use in this experiment.

## Features Meaning:

- `mtp` : Take the minimum of the probabilities that the LLM_E gives to the tokens on the generated-text.
- `avgtp` : Take the average of the probabilities that the LLM_E
gives to the tokens on the generated-text.
- `MDVTP` : Take the maximum from all the differences
between the token with the highest probability
according to LLM_E at position i and the
assigned probability from LLM_E to the token at position i in the generated_text.
- `MMDVP` : Take the maximum from all the differences between the token with the highest probability according to $LLM_E$ at position $i$ ($v^*$) and the token with the lowest probability according to $LLM_E$ at position $i$ ($v^-$).

"""

feature_to_extract = 'all'

available_features_to_extract = ["mtp", "avgtp", "MDVTP", "MMDVP"]
if feature_to_extract == 'all':
  features_to_extract = {
      feature: True for feature in available_features_to_extract
  }
else:
  features_to_extract = {
      feature: True if feature == feature_to_extract else False
      for feature in available_features_to_extract
  }

features_to_extract

"""## Cleaning Cache on GPU to save memory"""


gc.collect()
torch.cuda.empty_cache()

"""## This cell is to instantiate the model you intend to use for the experiment"""

#model = BartCNN()
#model = LED()
#model = GPT2Generator()
model = LLama()
#model = Gemma()
#model = Opt()
#model = Gptj()

"""## This cell creates the dataset separation of `10%` for training and `90%` for testing depending on what task you are addressing. The following explanation is what happens if summarization is the task used. But the same explanation applies to all tasks and also you cand pass as parameter how many data points you want to include in training.

## Example: The data is separated on 2000 (1000 of document with right summary and 1000 with the same document but with the hallucinated summary). The rest which is 18000 is used to for testing.

### As expected from previous cells the task string expected are:
- `summarization`
- `qa`
- `dialogue`
- `general`
"""


# Load the data depending on the task
def loadRowData(taskName, row, includeKnowledge=False):
  if taskName == "summarization":
    return "", row["document"], row["right_summary"], row["hallucinated_summary"]
  elif taskName == "qa":
    if includeKnowledge:
      return (
          row["knowledge"],
          row["question"],
          row["right_answer"],
          row["hallucinated_answer"],
      )
    else:
      return "", row["question"], row["right_answer"], row["hallucinated_answer"]

  elif taskName == "dialogue":
    if includeKnowledge:
      return (
          row["knowledge"],
          row["dialogue_history"],
          row["right_response"],
          row["hallucinated_response"],
      )
    else:
      return (
          "",
          row["dialogue_history"],
          row["right_response"],
          row["hallucinated_response"],
      )

  elif taskName == "general":
    return (
        "",
        row["user_query"],
        row["chatgpt_response"],
        row["hallucination_label"],
    )

  else:
    raise Exception("Task not supported")


# Adapt the dataset to have a data point of conditioned-text with
# right-generation and another with the same conditioned-text and tha
# hallucinated answer.
def adaptDataset(data, taskName):
  datasetAdapted = None
  if taskName == "general":
    # There is data point that is filling the <mask> token but that gives
    # error with some LLMs
    datasetAdapted = [
        (
            (knowledge, document, response, 1)
            if hallu == "yes"
            else (knowledge, document, response, 0)
        )
        for knowledge, document, response, hallu in data
        if "<mask>" not in document and "<mask>" not in response
    ]

  elif taskName == "summarization" or taskName == "qa" or taskName == "dialogue":
    datasetAdapted = [
        (knowledge, document, right, 1)
        for knowledge, document, right, hallu in data
    ] + [
        (knowledge, document, hallu, 0)
        for knowledge, document, right, hallu in data
    ]

  else:
    raise Exception("Task not supported")

  random.shuffle(datasetAdapted)
  return datasetAdapted


def splitDataset(
    data: pd.DataFrame,
    taskName: str,
    trainingSize: int,
    valSize: int,
    includeKnowledge=False,
):

  dataset = []
  for _, row in data.iterrows():
    knowledge, text, right, hallu = loadRowData(
        taskName, row, includeKnowledge)
    dataset.append((knowledge, text, right, hallu))

  random.shuffle(dataset)

  dataset_train = dataset[:trainingSize]  # Take only trainingSize
  dataset_val = (
      []
  )  # dataset[trainingSize:trainingSize + valSize] # Take only trainingSize
  dataset_test = dataset[trainingSize:]  # Take the rest as testing

  datasetAdaptedTrain = adaptDataset(dataset_train, taskName)
  datasetAdaptedValidation = adaptDataset(dataset_val, taskName)
  datasetAdaptedTest = adaptDataset(dataset_test, taskName)

  X_train = [(x, q, y) for x, q, y, _ in datasetAdaptedTrain]
  Y_train = [z for _, _, _, z in datasetAdaptedTrain]

  X_val = [(x, q, y) for x, q, y, _ in datasetAdaptedValidation]
  Y_val = [z for _, _, _, z in datasetAdaptedValidation]

  X_test = [(x, q, y) for x, q, y, _ in datasetAdaptedTest]
  Y_test = [z for _, _, _, z in datasetAdaptedTest]

  return X_train, Y_train, X_val, Y_val, X_test, Y_test


includeKnowledge = True
includeConditioned = True

X_train, Y_train, X_val, Y_val, X_test, Y_test = splitDataset(
    data, task, 1000, 0, includeKnowledge=includeKnowledge
)

print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))
print(len(X_test), len(Y_test))  # verify the sizes look right

"""## To Save the separation if needed"""

train_df = pd.DataFrame(
    {
        "Knowledge": [x[0] for x in X_train],
        "Conditioned Text": [x[1] for x in X_train],
        "Generated Text": [x[2] for x in X_train],
        "Label": Y_train,
    }
)

val_df = pd.DataFrame(
    {
        "Knowledge": [x[0] for x in X_val],
        "Conditioned Text": [x[1] for x in X_val],
        "Generated Text": [x[2] for x in X_val],
        "Label": Y_val,
    }
)

test_df = pd.DataFrame(
    {
        "Knowledge": [x[0] for x in X_test],
        "Conditioned Text": [x[1] for x in X_test],
        "Generated Text": [x[2] for x in X_test],
        "Label": Y_test,
    }
)

# Export to CSV
if includeKnowledge:
  train_df.to_csv(
      output_path / (task + '_knowledge_train_data.csv'),
      index=False)
  test_df.to_csv(
      output_path / (task + '_knowledge_test_data.csv'),
      index=False)
else:
  train_df.to_csv(output_path / (task + '_train_data.csv'), index=False)
  val_df.to_csv(output_path / (task + '_val_data.csv'), index=False)
  test_df.to_csv(output_path / (task + '_test_data.csv'), index=False)


def getXY(df: pd.DataFrame, includeKnowledge=True, includeConditioned=True):
  X = []
  Y = []

  # Iterate over rows using itertuples
  for _, row in df.iterrows():
    x, c, g = (
        row["Knowledge"] if includeKnowledge else "",
        row["Conditioned Text"] if includeConditioned else "",
        row["Generated Text"],
    )
    y = row["Label"]

    # Append values to respective lists
    X.append((x, c, g))
    Y.append(y)
  return X, Y


X_train, Y_train = getXY(
    train_df, includeKnowledge=includeKnowledge,
    includeConditioned=includeConditioned)
X_val, Y_val = getXY(
    val_df, includeKnowledge=includeKnowledge,
    includeConditioned=includeConditioned)
X_test, Y_test = getXY(
    test_df, includeKnowledge=includeKnowledge,
    includeConditioned=includeConditioned)

print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))
print(len(X_test), len(Y_test))  # verify the sizes look right

X_test[0]

Y_test[0]

"""## Extracting the features for the Training Data"""


def extract_features(
    knowledge: str,
    conditioned_text: str,
    generated_text: str,
    features_to_extract: dict[str, bool],
):
  return model.extractFeatures(
      knowledge, conditioned_text, generated_text, features_to_extract
  )


X_train_features_maps = []

for knowledge, conditioned_text, generated_text in tqdm(
        X_train, desc="Processing"):
  X_train_features_maps.append(
      extract_features(
          knowledge, conditioned_text, generated_text, features_to_extract
      )
  )
  torch.cuda.empty_cache()  # Clean cache in every step for memory saving.

len(X_train_features_maps)

X_train_features_maps[0]

X_train_features = [list(dic.values()) for dic in X_train_features_maps]

len(X_train_features)

X_train_features[0]

"""## Training Logistic Regression"""


clf = LogisticRegression(verbose=1)
clf.fit(X_train_features, Y_train)

"""## Evaluate accuracy of Logistic Regression on the training set"""


Y_Pred = clf.predict(X_train_features)

accuracy = accuracy_score(Y_train, Y_Pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

log_odds = clf.coef_[0]
odds = np.exp(clf.coef_[0])
lr_features_log = {
    k: v for k,
    v in zip(
        X_train_features_maps[0].keys(),
        log_odds)}
lr_features_no_log = {
    k: v for k, v in zip(
        X_train_features_maps[0].keys(), odds)}

print("log", lr_features_log)
print("no_log", lr_features_no_log)

"""## Extracting the Features of the Validation Set"""

X_val_features_map = []

for knowledge, conditioned_text, generated_text in tqdm(
        X_val, desc="Processing"):
  X_val_features_map.append(
      extract_features(
          knowledge, conditioned_text, generated_text, features_to_extract
      )
  )
  torch.cuda.empty_cache()

X_val_features = [list(dic.values()) for dic in X_val_features_map]

"""## Uncomment nex cell if you have a validation set and you want to see LR accuracy on it."""

# from sklearn.metrics import accuracy_score

# Y_Pred = clf.predict(X_val_features)

# accuracy = accuracy_score(Y_val, Y_Pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

"""## Extracting the Features of the Test Set"""


X_test_features_map = []

for knowledge, conditioned_text, generated_text in tqdm(
        X_test, desc="Processing"):
  X_test_features_map.append(
      extract_features(
          knowledge, conditioned_text, generated_text, features_to_extract
      )
  )
  torch.cuda.empty_cache()

X_test_features = [list(dic.values()) for dic in X_test_features_map]

"""## Evaluate accuracy of the LogisticRegression on the testing set"""


Y_Pred = clf.predict(X_test_features)

lr_accuracy = accuracy_score(Y_test, Y_Pred)
print(f"Accuracy: {lr_accuracy * 100:.2f}%")

log_odds = clf.coef_[0]
pd.DataFrame(
    log_odds,
    X_train_features_maps[0].keys(),
    columns=["coef"]).sort_values(
    by="coef",
    ascending=False)

odds = np.exp(clf.coef_[0])
pd.DataFrame(
    odds, X_train_features_maps[0].keys(),
    columns=["coef"]).sort_values(
    by="coef", ascending=False)


class SimpleDenseNet(nn.Module):
  def __init__(
          self,
          input_dim: int,
          hidden_dim: int,
          output_dim=1,
          dropout_prob=0.3):
    super(SimpleDenseNet, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, output_dim)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.sigmoid(x)
    return x


denseModel = SimpleDenseNet(
    input_dim=np.array([v for v in features_to_extract.values()]).sum(),
    hidden_dim=512).to(device)

"""# Code declaring and computing all the metrics to measure"""


def compute_metrics(model, input_tensor, true_labels):
  with torch.no_grad():
    outputs = model(input_tensor)
    predicted_probs = torch.sigmoid(outputs).cpu().numpy()
    predicted = (outputs > 0.5).float().cpu().numpy()

    true_labels = true_labels.cpu().numpy()

    acc = accuracy_score(true_labels, predicted)
    precision = precision_score(true_labels, predicted)
    recall = recall_score(true_labels, predicted)
    f1 = f1_score(true_labels, predicted)

    precision_negative = precision_score(true_labels, predicted, pos_label=0)
    recall_negative = recall_score(true_labels, predicted, pos_label=0)
    f1_negative = f1_score(true_labels, predicted, pos_label=0)

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted).ravel()
    roc_auc = roc_auc_score(true_labels, predicted_probs)

    P, R, thre = precision_recall_curve(true_labels, predicted, pos_label=1)
    pr_auc = auc(R, P)

    roc_auc_negative = roc_auc_score(
        true_labels, 1 - predicted_probs
    )  # If predicted_probs is the probability of the positive class
    P_neg, R_neg, _ = precision_recall_curve(
        true_labels, predicted, pos_label=0)
    pr_auc_negative = auc(R_neg, P_neg)

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Precision-Negative": precision_negative,
        "Recall-Negative": recall_negative,
        "F1-Negative": f1_negative,
        "ROC AUC-Negative": roc_auc_negative,
        "PR AUC-Negative": pr_auc_negative,
    }


"""## Code for training the Dense Model and getting the result of all metrics corresponding to the Testing Set."""


def compute_accuracy(model, input_tensor, true_labels):
  with torch.no_grad():
    outputs = model(input_tensor)
    predicted = (outputs > 0.5).float()
    correct = (predicted == true_labels).float().sum()
    accuracy = correct / len(true_labels)
    return accuracy.item()


X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(
    Y_train, dtype=torch.float32).view(-1, 1).to(device)

print(X_train_tensor.shape, Y_train_tensor.shape)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(denseModel.parameters(), lr=0.001)

bestValAcc = 0
# Training loop
num_epochs = 20000
for epoch in range(num_epochs):
  denseModel.train()
  optimizer.zero_grad()
  outputs = denseModel(X_train_tensor)
  loss = criterion(outputs, Y_train_tensor)
  loss.backward()
  optimizer.step()

  # Compute training accuracy
  train_accuracy = compute_accuracy(denseModel, X_train_tensor, Y_train_tensor)

  # Uncomment this if you want to see how the accuracy of testing improves during the training process.
  # Compute testing accuracy
  # X_val_tensor = torch.tensor(X_val_features, dtype=torch.float32).to(device)
  # Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1).to(device)

  # val_accuracy = compute_accuracy(denseModel, X_val_tensor, Y_val_tensor)

  # if bestValAcc < val_accuracy:
  #     bestValAcc = val_accuracy
  #     print(f'Saving model with best validation accuracy ...')
  #     torch.save(denseModel.state_dict(), 'llama-' + task + '-best-model')

  if (epoch + 1) % 10 == 0:
    print(
        f"Epoch [{
            epoch + 1}/{num_epochs}], Loss: {
            loss.item():.4f}, Training Accuracy: {
            train_accuracy:.4f}")  # , "Validation Accuracy": {val_accuracy:.4f}')

"""## Uncomment next cell if you want to load a particular model you already trained."""

# loaded_model = SimpleDenseNet(input_dim=len(list(features_to_extract.keys())), hidden_dim=512).to(device)
# loaded_model.load_state_dict(torch.load('llama-' + task + '-best-model'))

# # Set the model to evaluation mode"
# loaded_model.eval()

"""#Compute the metrics using the model on the Test Set."""

X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(
    Y_test, dtype=torch.float32).view(-1, 1).to(device)

test_metrics = compute_metrics(denseModel, X_test_tensor, Y_test_tensor)

print(
    f"Testing - Accuracy: {
        test_metrics['Accuracy']:.4f}, Precision: {
        test_metrics['Precision']:.4f}, Recall: {
        test_metrics['Recall']:.4f}, F1: {
        test_metrics['F1']:.4f}, ROC AUC: {
            test_metrics['ROC AUC']:.4f}, PR AUC: {
        test_metrics['PR AUC']:.4f}")
print(
    f"Testing - Negative: {
        test_metrics['Accuracy']:.4f}, Precision-Negative: {
        test_metrics['Precision-Negative']:.4f}, Recall-Negative: {
        test_metrics['Recall-Negative']:.4f}, F1-Negative: {
        test_metrics['F1-Negative']:.4f}, ROC AUC-Negative: {
            test_metrics['ROC AUC-Negative']:.4f}, PR AUC-Negative: {
        test_metrics['PR AUC-Negative']:.4f}")

"""## Save the results on a CSV if you want."""

model_dataframe = pd.DataFrame(
    columns=[
        "features",
        "model_name",
        "feature_to_extract",
        "method",
        "accuracy",
        "precision",
        "recall",
        "roc auc",
        "pr auc",
        "negative",
        "precision-negative",
        "recall-negative",
        "negative f1",
        "lr_accuracy",
        "lr_features_log",
        "lr_features_no_log",
    ]
)

d = {
    "features": features_to_extract,
    "model_name": str(model.getName()),
    "feature_to_extract": feature_to_extract,
    "method": "TEST",
    "accuracy": test_metrics["Accuracy"],
    "precision": test_metrics["Precision"],
    "recall": test_metrics["Recall"],
    "f1": test_metrics["F1"],
    "pr auc": test_metrics["PR AUC"],
    "precision-negative": test_metrics["Precision-Negative"],
    "recall-negative": test_metrics["Recall-Negative"],
    "negative-f1": test_metrics["F1-Negative"],
    "lr_accuracy": lr_accuracy,
    "lr_features_log": lr_features_log,
    "lr_features_no_log": lr_features_no_log,
}

model_dataframe.loc[len(model_dataframe.index)] = d

model_dataframe.head()

csv_name = f"{
    model.getSanitizedName()} _{task} _{
    includeKnowledge=} _{
    includeConditioned=} _{
    '_'.join([f'{k}={v} 'for k, v in features_to_extract.items()])}.csv"
model_dataframe.to_csv(output_path / csv_name, index=False)
