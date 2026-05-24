corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

from collections import defaultdict

''' logic step: 
1.we want to caculate the frequency of each pair of tokens in the corpus
2.first find out the unique token frequency
3.then find out the frequency of each pair of tokens

'''

# auto add up the frequency of each token in the corpus
word_freq = defaultdict(int)

for sentence in corpus:
    for words in sentence.split():
        word_freq[words] += 1

print(word_freq)

# find the frequency of each pair of tokens
pair_freq = defaultdict(int)

for words in word_freq:
    if len(words) == 1:
        continue
    for i in range(len(words)-1):
        pair_freq[words[i],words[i+1]] += word_freq[words]


print(pair_freq)


# find the most frequent pair of tokens


def find_best_pair(pair_freq, max_freq=None, best_pair=""):
  for pair, freq in pair_freq.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq
  return best_pair, max_freq


# merge the most frequent pair of tokens into a single token and update the frequency of the new token and the pairs that contain the new token

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


vocab = dict()

def merge_pair(a, b, splits):
    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits



vocab_size = 50

while len(pair_freqs) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])




def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
