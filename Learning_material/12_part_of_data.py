
#----------------------------------------------------------
#
#  This file is part of the HuggingFace course, created by Sylvain Gugger and the HuggingFace team.
#  This course is to process the pretrained data, like remove the None value, add new column, etc.
#----------------------------------------------------------


# tsv is conversion of csv, so we can use csv to load tsv file

from datasets import load_dataset
from transformers import AutoTokenizer

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}


drug_dataset = load_dataset("csv", data_files=data_files, 
                            # specify the delimiter for tsv file
                            delimiter="\t")

#start slice

drug_dataset_samples = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_dataset_samples[:3])


for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique('Unnamed: 0'))


drug_dataset = drug_dataset.rename_column("Unnamed: 0", "id")


'''


DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})



'''


# turn every condition into lowercase

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


# filter out the examples where condition is None
def filter_nones(x):
    return x["condition"] is not None


drug_dataset = drug_dataset.filter(filter_nones)
drug_dataset = drug_dataset.map(lowercase_condition)

# use mamba to filter the attribure

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)



# add new column

def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

# use map to add new column review_length
drug_dataset = drug_dataset.map(compute_review_length)



# delete the column 
drug_dataset = drug_dataset.filter(lambda x: x["review_length"]>30)



# convert the uncode to normal text, by using html.unescape

import html

text = "I&#039;m a transformer called BERT"

# use html to unescape the text
html.unescape(text)


drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})


# use map with batched=True to unescape the review column in batches

new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
    
)


# --------------------------------------------------

# comparison using whether fast or slow tokenizer

slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

'''

本质区别

    Slow Tokenizer：Python 实现，基于 HuggingFace 的 tokenizers 库的 Python 绑定

    Fast Tokenizer：Rust 实现，基于同一个库但完全用 Rust 编写，通过 Python 绑定调用

主要差异对比
特性	Slow Tokenizer	Fast Tokenizer
实现语言	Python	Rust
速度	慢	快（5-10倍）
功能	基础功能	完整功能
返回信息	仅 input_ids	额外返回 offset_mapping、word_ids 等
批处理	手动处理	原生支持
内存占用	较高	较低

'''
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)






def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

