from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# training argment

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

'''
training_args = TrainingArguments(

    output_dir="test-trainer",  # 输出目录
    # 以下都是默认值
    num_train_epochs=3,              # 训练3个epoch
    per_device_train_batch_size=8,   # 每个设备的训练批次大小
    per_device_eval_batch_size=8,    # 每个设备的评估批次大小
    warmup_steps=0,                  # 预热步数
    weight_decay=0,                  # 权重衰减
    logging_dir=None,                # 日志目录
    logging_steps=500,               # 每500步记录一次日志
    save_steps=500,                  # 每500步保存一次检查点
    evaluation_strategy="no",        # 评估策略（不评估）
    # ... 还有很多其他参数

)
'''


from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)



from transformers import Trainer

trainer = Trainer(
    model,
    # training argument
    training_args,

    #train and evaluation dataset
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],


    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train()

# this will only output 