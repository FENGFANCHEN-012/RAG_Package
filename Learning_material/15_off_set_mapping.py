from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain"

# 返回偏移映射
encoding = tokenizer(example, return_offsets_mapping=True)

print(encoding["offset_mapping"])
# [(0, 0), (0, 2), (3, 7), (8, 10), (11, 12), (12, 14), (14, 16), (16, 18)]

# 特殊 token [CLS] 的偏移是 (0, 0)
# "My" → (0, 2)
# "name" → (3, 7)
# "is" → (8, 10)
# "S" → (11, 12)
# "##yl" → (12, 14)
# "##va" → (14, 16)
# "##in" → (16, 18)



# 问题：token 可能不是完整的单词
tokens = encoding.tokens()
print(tokens)
# ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in']

# 如何知道 "S", "##yl", "##va", "##in" 实际上组成 "Sylvain"？
# 答案：通过偏移映射！

start, end = encoding.word_to_chars(4)  # 获取第4个token对应的字符范围
print(example[start:end])  # 输出: "Sylvain"


#  
encoding = tokenizer("Hello world")
print(type(encoding))  # BatchEncoding

# 快速 tokenizer 特有的方法
encoding.tokens()           # 获取所有 token
encoding.word_ids()         # 每个 token 对应的单词索引
encoding.sentence_ids()     # 每个 token 对应的句子索引
encoding.word_to_chars()    # 单词映射到字符位置
encoding.char_to_token()    # 字符位置映射到 token


# 1. 获取偏移映射
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)

# 2. 获取模型预测
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# 3. 使用偏移映射定位实体在原文中的位置
for idx, pred in enumerate(predictions):
    if pred != "O":  # 不是非实体
        start, end = offsets[idx]  # 获取在原文中的位置
        entity_text = example[start:end]  # 提取实体文本


# 错误方法：需要为不同 tokenizer 编写不同规则
if tokenizer_type == "BERT":
    # 处理 ## 前缀
elif tokenizer_type == "GPT2":
    # 处理 Ġ 前缀
elif tokenizer_type == "SentencePiece":
    # 处理 _ 前缀

# 正确方法：使用偏移映射（通用）
start, end = offsets[first_token_idx]  # 第一个 token 的开始
_, end = offsets[last_token_idx]       # 最后一个 token 的结束
full_entity = example[start:end]       # 直接提取完整实体