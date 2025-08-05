!pip install transformers datasets

pip install transformers datasets scikit-learn pandas openpyxl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google.colab import files

uploaded = files.upload()

import pandas as pd

# Load the uploaded Excel files
annotated_df = pd.read_excel("cool.annotated.filtered.cleaned.xlsx")
unannotated_df = pd.read_excel("cool.unannotated.filtered.xlsx")

# Optional: preview a few rows
annotated_df.head(), unannotated_df.head()

from sklearn.preprocessing import LabelEncoder

# Clean whitespace and inconsistent capitalization
annotated_df["interpretation"] = annotated_df["interpretation"].str.strip().str.capitalize()

# Encode the cleaned labels into numeric values
label_encoder = LabelEncoder()
annotated_df["label"] = label_encoder.fit_transform(annotated_df["interpretation"])

# Optional: Show the mapping from label names to integers
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Load pretrained tokenizer and model (3 labels for classification)
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

from sklearn.model_selection import train_test_split
from datasets import Dataset

# Split into training and test sets
train_df, test_df = train_test_split(annotated_df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format (renaming 'occurrences' → 'text')
train_dataset = Dataset.from_pandas(train_df[["occurrences", "label"]].rename(columns={"occurrences": "text"}))
test_dataset = Dataset.from_pandas(test_df[["occurrences", "label"]].rename(columns={"occurrences": "text"}))

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Format datasets for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./cool_model_xlm",
    eval_strategy="epoch",  # Evaluate at each epoch
    save_strategy="epoch",  # Save at each epoch
    load_best_model_at_end=True,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "xlm-roberta-base" # Or any other model you want to use
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # Adjust num_labels as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Tokenize both train and test datasets using your tokenizer
tokenized_train = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding="max_length"), batched=True)
tokenized_test = test_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding="max_length"), batched=True)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)
# Start training
trainer.train()
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Evaluate the model on the test set
eval_results = trainer.evaluate()

# Print the evaluation results
print(eval_results)
pip install evaluate scikit-learn
# Import necessary libraries
import numpy as np
import evaluate

# Define the function that will compute the metrics
def compute_metrics(eval_pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.
    """
    # The 'eval_pred' object is a tuple containing the model's raw output (logits)
    # and the true labels.
    logits, labels = eval_pred

    # The logits are the raw scores. To get the final prediction, we take the
    # class with the highest score (the argmax).
    predictions = np.argmax(logits, axis=-1)

    # Load the metric calculators from the 'evaluate' library
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    # Calculate the scores
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    # For multi-class metrics, 'average="weighted"' accounts for class imbalance.
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")

    # Return the results as a dictionary
    return {
        "accuracy": accuracy['accuracy'],
        "f1": f1['f1'],
        "precision": precision['precision'],
        "recall": recall['recall']
    }
# Start training
trainer.train()
from datasets import Dataset

# Prepare the unannotated dataframe for tokenization
unannotated_dataset = Dataset.from_pandas(
    unannotated_df[["match_context"]].rename(columns={"match_context": "text"})
)

# Tokenize the text
tokenized_unannotated = unannotated_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length"),
    batched=True
)

# Format for PyTorch
tokenized_unannotated.set_format("torch", columns=["input_ids", "attention_mask"])

# Run predictions
predictions = trainer.predict(tokenized_unannotated)

# Get predicted class index (highest score)
predicted_class_ids = predictions.predictions.argmax(axis=1)

# Convert numeric predictions back to label names
predicted_labels = label_encoder.inverse_transform(predicted_class_ids)

# Add predictions to the unannotated DataFrame
unannotated_df["predicted_interpretation"] = predicted_labels

# Save to Excel
unannotated_df.to_excel("cool.unannotated.with_predictions.xlsx", index=False)

# Confirmation message
print("✅ Predictions saved to: cool.unannotated.with_predictions.xlsx")

from google.colab import files
files.download("cool.unannotated.with_predictions.xlsx")

import numpy as np

# Get full probability scores (logits → softmax to get probabilities)
from scipy.special import softmax
probabilities = softmax(predictions.predictions, axis=1)

# Create a score column: confidence value of the predicted class
prediction_scores = np.max(probabilities, axis=1)  # max prob = confidence score

# Convert class indices to label names
predicted_labels = label_encoder.inverse_transform(predictions.predictions.argmax(axis=1))

# Add both to DataFrame
unannotated_df["predicted_interpretation"] = predicted_labels
unannotated_df["prediction_score"] = prediction_scores

unannotated_df.to_excel("cool.unannotated.with_predictions.xlsx", index=False)
print("✅ File updated with prediction scores.")


from google.colab import files
files.download("cool.unannotated.with_predictions.xlsx")
