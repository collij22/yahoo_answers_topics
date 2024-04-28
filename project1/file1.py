from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = load_dataset("yelp_review_full")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.remove_columns_("text")
tokenized_dataset.rename_column_("label", "labels") # rename the column label to labels to match the expected name for the labels
tokenized_dataset.set_format("torch") #convert the dataset to PyTorch tensors
small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
small_test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(2000))



