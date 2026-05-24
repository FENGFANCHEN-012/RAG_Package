import requests
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login


GITHUB_TOKEN = "你的token"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

all_issues = []
for page in range(5):  
    url = f"https://api.github.com/repos/huggingface/datasets/issues?page={page}&per_page=100&state=all"
    response = requests.get(url, headers=headers)
    all_issues.extend(response.json())

# 2. 保存为 JSON Lines
df = pd.DataFrame.from_records(all_issues)
df.to_json("my-dataset.jsonl", orient="records", lines=True)

# 3. 加载到 🤗 Datasets
dataset = load_dataset("json", data_files="my-dataset.jsonl", split="train")

# 4. 清洗和扩充

def add_label(example):
    return {"is_issue": example["pull_request"] is None}

dataset = dataset.map(add_label)

# 5. 上传
notebook_login()
dataset.push_to_hub("my-github-issues")