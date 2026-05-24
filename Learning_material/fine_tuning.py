from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.optim import AdamW


checkpoint = "distillgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)


sequence = [
    "What are your answer if i told you i am a man? I used to save a boy",
    "I am really good person? ","Life gonna destory me? "]
 
 
# return tensor if it is py
# return gonna be a dictionary with input_ids and attention_mask
model_input = tokenizer(sequence,padding = "max_length",truncation = True,
                         max_length = 512, return_tensors="pt")


# set label, in real world, you should have your own label for your dataset, here we just set a random label for demonstration
model_input["labels"] = model_input["input_ids"].clone()


optimizer = AdamW(model.parameters(), lr=5e-5)


model.train()

# in transformers, forward function only accept inputids, attention_mask and labels, so we need to unpack the dictionary
outputs = model(**model_input)
loss = outputs.loss
print(loss)
loss.backward()





