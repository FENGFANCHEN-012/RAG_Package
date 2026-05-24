from transformers import AutoTokenizer
from datasets import load_dataset


# data set can choose train
dataset = load_dataset("code_search_net", "python", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i:i+1000]["whole_func_string"]


# 3. 基于 GPT-2 的 tokenizer 架构训练新 tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = old_tokenizer.train_new_from_iterator(
    get_training_corpus(), 
    vocab_size=52000  # 新词汇表大小
)

# 4. 保存
new_tokenizer.save_pretrained("my-code-tokenizer")

# 5. 上传到 Hub（可选）
new_tokenizer.push_to_hub("your-username/code-tokenizer")