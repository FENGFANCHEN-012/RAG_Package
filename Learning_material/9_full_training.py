
# 1.import tokenizer model and load the dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithFlattening


checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)


from datasets import load_dataset
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")    


# 2. tokenize the dataset and prepare the data for training

def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
data_collator = DataCollatorWithFlattening(tokenizer=tokenizer, mlm=False)



# remove the column

'''
 假设 tokenized_datasets 包含这些列：
{
    'input_ids': [...],      # 模型需要的
    'attention_mask': [...], # 模型需要的
    'label': [...],          # 模型需要的
    'sentence1': "Hello",    # ❌ 模型不认识，会报错！
    'sentence2': "World",    # ❌ 模型不认识，会报错！
    'idx': 0                 # ❌ 模型不认识，会报错！
}

'''
# 3. remove the columns that are not needed for training, rename the label column to labels, and set the format to torch
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names


# prepare the data for training

from torch.utils.data import DataLoader


# truffle the data into batches and collate them into the same length
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


# Model for classification

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}


outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)



# decrease loss
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)



# use GPU to train

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)




#
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# evaluate the model


import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()