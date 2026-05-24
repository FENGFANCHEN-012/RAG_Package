
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
token = tokenizer.tokenize("Hi, I am a gay, do you want to have a chat with me?")


# convert token to id
token_id = tokenizer.convert_tokens_to_ids(token)
print(token_id)


# decoding

decode_token = tokenizer.convert_ids_to_tokens(token_id)
print(decode_token)


# import a model which is for classification
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text = "i am so glad to see you again"

encode_text  = tokenizer.tokenize(text)
encode_text_id = tokenizer.convert_tokens_to_ids(encode_text)

# there should have one more dimension for tensor 
input_id = torch.tensor([encode_text_id])

# model only accpet tensor as input, so we need to convert the list to tensor
output = model(input_id)


# batch example

batch_ids = [[
    1,2,3
],[2,3,tokenizer.pad_token_id]]


print(model(input_id).logits)


# attention mask

attention_mask = [[1,1,1],[1,1,0]]

output = model(input_id,attention_mask=torch.tensor(attention_mask))



