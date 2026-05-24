'''
┌─────────────────────────────────────────────────────────┐
│                    Tokenizer 类                          │
├─────────────────────────────────────────────────────────┤
│ 1. Normalizer    - 文本标准化（转小写、去重音等）          │
│ 2. PreTokenizer  - 预分词（按空格、标点切分）             │
│ 3. Model         - 核心分词算法（BPE/WordPiece/Unigram） │
│ 4. Trainer       - 训练器（在语料上训练词汇表）           │
│ 5. PostProcessor - 后处理（添加特殊 tokens）             │
│ 6. Decoder       - 解码器（将 token IDs 转回文本）       │
└─────────────────────────────────────────────────────────┘


'''

from datasets import load_dataset

# 加载 WikiText-2 数据集（原始文本）
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

# 创建生成器，每次返回 1000 个文本
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]



from tokenizers import Tokenizer, models

# 创建 Tokenizer，使用 WordPiece 模型
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# unk_token: 未知词标记，当遇到没见过的字符时返回这个


from tokenizers import normalizers

# 方式1：使用 BERT 预置的 normalizer
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

# 方式2：手动构建（更灵活）
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),           # Unicode 规范化
    normalizers.Lowercase(),     # 转小写
    normalizers.StripAccents()   # 去除重音符号
])

# 测试
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
# 输出: "hello how are u?"


# STEP 3：预分词器（PreTokenizer）
from tokenizers import pre_tokenizers

# 使用 BERT 风格的预分词（按空格和标点切分）
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# 或者手动构建
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),  # 先按空格切分
    pre_tokenizers.Punctuation()       # 再处理标点
])

# 测试
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test!"))
# 输出: [('Let', (0,3)), ("'", (3,4)), ('s', (4,5)), ('test', (6,10)), ('!', (10,11))]


# STEP 4：核心分词算法（Model）

from tokenizers import trainers

# 定义特殊 tokens
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# 创建训练器
trainer = trainers.WordPieceTrainer(
    vocab_size=25000,           # 词汇表大小
    special_tokens=special_tokens,  # 特殊 tokens
    min_frequency=2             # 最小出现频率（可选）
)

# 使用生成器训练
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# 或使用文件训练
# tokenizer.train(["wikitext-2.txt"], trainer=trainer)



# STEP 5: 测试结果

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
# 输出: ['let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.']
# 注意: '##' 表示这是子词的延续

# STEP 6: 后处理器（PostProcessor）
from tokenizers import processors

# 获取特殊 token 的 ID
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

# 设置模板
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",                    # 单句模板
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",        # 双句模板
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
)

# 测试
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
# 输出: ['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']


# STEP 7: 解码器（Decoder）

from tokenizers import decoders

tokenizer.decoder = decoders.WordPiece(prefix="##")

# 测试解码
print(tokenizer.decode(encoding.ids))
# 输出: "let's test this tokenizer."


# STEP 8: 保存

# 保存 tokenizer
tokenizer.save("tokenizer.json")

# 封装到 Transformers
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 现在可以像标准 tokenizer 一样使用
wrapped_tokenizer.save_pretrained("my_tokenizer")


