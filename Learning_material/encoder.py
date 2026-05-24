from transformers import BertConfig, BertModel

config = BertConfig()

model = BertModel(config)


# tokenizer for bert-base-cased
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# bert tokenizer only for bert model, ier model

# tokenizer for auto model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# save the tokenizer to a directory on my computer
tokenizer.save_pretrained("directory_on_my_computer")



sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)


# convert tokens to ids
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

