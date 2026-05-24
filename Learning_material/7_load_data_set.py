

from datasets import load_dataset
from transformers import AutoTokenizer



raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# dataset will become a dictionary with train, test and validation set
'''
DatasetDict({

    train: Dataset({
      
    'sentence1': ['文本1', '文本2', '文本3', ...],  # 3668个元素的列表/数组
    'sentence2': ['文本1', '文本2', '文本3', ...],  # 3668个元素的列表/数组
    'label': [0, 1, 0, ...],                       # 3668个元素的列表/数组
    'idx': [0, 1, 2, ...]                          # 3668个元素的列表/数组

    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
'''

# find out the train dataset in raw_datasets, and print the first element in the train dataset
raw_train_dataset = raw_datasets["train"]

length_of_each_feaure = []

for i in range(len(raw_train_dataset)):
    tokenlength = len(tokenizer(raw_train_dataset[i]['sentence1'], raw_train_dataset[i]['sentence2'],truncation=True,padding=False)
                      ["input_ids"])
    length_of_each_feaure.append(tokenlength)

# Length of each feature in train is different

print(length_of_each_feaure)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


