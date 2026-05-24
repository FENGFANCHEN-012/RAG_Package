from datasets import load_dataset

data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")


# 
import psutil

print(f"RAM: Used {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")



print(f"数据集中文件的数量 : {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"数据集大小 (缓存文件) : {size_gb:.2f} GB")




import timeit

# check how long it takes to iterate through the dataset

code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())


print(
    f"在 {time:.1f}s 内遍历了 {len(pubmed_dataset)}个示例(约 {size_gb:.1f} GB),即 {size_gb/time:.3f} GB/s"
)
