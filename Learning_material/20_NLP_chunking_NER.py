from datasets import load_dataset
raw_datasets = load_dataset("conll2003")


raw_datasets["train"][0]["tokens"]

# check the NER tags
raw_datasets["train"][0]["ner_tags"]


ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature.names



'''
O 表示这个词不对应任何实体。
B-PER / I-PER 意味着这个词对应于人名实体的开头/内部。
B-ORG / I-ORG 的意思是这个词对应于组织名称实体的开头/内部。
B-LOC / I-LOC 指的是是这个词对应于地名实体的开头/内部。
B-MISC / I-MISC 表示这个词对应一个其他实体（不属于特定类别或类别之外）的开头 / 内部。

'''

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

inputs = tokenizer(raw_datasets["train"][0]["tokens"])

# must add is_split_into_words=True when the input is a list of wordss
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()

'''
['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']

'''

inputs.word_ids()


'''

[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]

'''


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 新单词的开始!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊的token
            new_labels.append(-100)
        else:
            # 与前一个 tokens 类型相同的单词
            label = labels[word_id]
            # 如果标签是 B-XXX 我们将其更改为 I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

    
