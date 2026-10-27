import os, pickle
import regex as re
from collections import Counter

corpus_path = "D:/C++/LocalAI/code_corpus.txt"
out_dir = "D:/C++/LocalAI/code_artifacts"
os.makedirs(out_dir, exist_ok=True)

# Read corpus
with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Pre-tokenize into words
words = re.findall(pattern, text)
print("num words", len(words))

# Word frequency
word_freq = Counter(words)
# Convert each word to tuple of byte symbols
def word_to_symbols(w):
    return tuple(bytes([b]) for b in w.encode("utf-8"))

# Build symbol sequences per unique word
word_symbols = {}
for w, freq in word_freq.items():
    word_symbols[w] = (word_to_symbols(w), freq)

# Vocab: single bytes -> id, plus specials
# Vocab: single bytes -> id, plus specials
# special token ids as in ncm: special_token_ids = {token: idx for idx, token in enumerate(special_tokens.values())}
# For tinystories, special_tokens = {'eos_token': ' authDomain', 'pad_token': '<|pad|>'} with eos first -> id 0? Actually special_token_ids maps token string -> idx based on enumerate(special_tokens.values()). 
# Let's define special_tokens dict matching config: eos_token and pad_token
special_tokens_cfg = {"eos_token": "\ue000", "pad_token": "<|pad|>"}

# Build vocab mapping symbol(byte-tuple) -> id
vocab = {}
# specials: map special token string -> id via their byte? In ncm, special tokens are stored as strings in special_tokens dict, and vocab maps byte-tuples. 
# But encode() handles special tokens separately (checks `if word in special_tokens`). And reverse_vocab maps id -> byte-tuple. For special tokens, vocab[id] is the special token's utf-8 bytes? Let's check: in encode, new_words appends `word.encode("utf-8")` for special tokens (as bytes). Then returns reverse_vocab[byte_pair] where byte_pair is bytes. So special tokens must be in vocab as their utf-8 bytes tuple? Actually word.encode("utf-8") is bytes; then [w for w in new_word] where new_word is tuple of bytes chunks. For special token, new_words.append(word.encode("utf-8")) (a bytes object), then `new_words += [w for w in new_word]`? Wait new_word for special is the bytes object; iterating gives ints, not bytes. Hmm, let's re-read.

# In encode:
#   if word in special_tokens: new_words.append(word.encode("utf-8"))  -> appends a bytes object to new_words list
# Then after loop: new_words += [w for w in new_word] for non-special. For special, it already appended. So new_words contains bytes objects for special tokens.
# Then return [reverse_vocab[byte_pair] for byte_pair in new_words]. So reverse_vocab must map bytes object -> id. But bytes object is not a tuple of bytes. Hmm.

# Actually reverse_vocab = {value: key for key, value in vocab.items()} where vocab is symbol->id. So vocab keys are the symbols (bytes objects or tuples of bytes). reverse_vocab maps id->symbol. So for special tokens, the symbol stored in vocab should be the full bytes object (word.encode("utf-8")), not a tuple.

# This is getting complicated. Let's simplify: store special tokens in vocab as their utf-8 bytes (as bytes object), and regular symbols as tuples of single-byte bytes objects (as ncm does for merged). Wait ncm's merge produces `earliest_merge[0] + earliest_merge[1]` which is bytes+bytes = bytes (concatenation). And parse_word produces tuple of bytes([b]). So symbols are either bytes objects (length 1, a single byte) or bytes objects (merged, length>1). Actually bytes([b]) is a bytes object of length 1. And merged is bytes (concatenation). So all symbols are bytes objects, not tuples! But code uses tuple([bytes([byt])...]) and tuple comparisons. Let me re-check.

# parse_word = lambda word: tuple(bytes([byt]) for byt in word)  -> tuple of bytes objects.
# merge: tmp.append(earliest_merge[0] + earliest_merge[1]) -> bytes object. So after merge, new_word becomes a mix: some elements bytes (merged), some bytes (single). Actually earliest_merge[0] is a bytes object (single or merged), + concatenates -> bytes. So symbols are bytes objects. But new_word is a tuple of these bytes objects. And _get_byte_pairs operates on tuple of bytes objects. candidate_merge is tuple of two bytes objects. merges keys are tuples of bytes objects.

# vocab keys: in encode final, `new_words` contains bytes objects (single merged units) for regular, and bytes objects for special. Then reverse_vocab[byte_pair] where byte_pair is a bytes object. So vocab keys are bytes objects.

# So vocab: bytes -> id. merges: dict mapping (bytes, bytes) -> rank.

# For special tokens: vocab[word.encode("utf-8")] = id.

# Good. So I'll train BPE where symbols are bytes objects.

# Let's redo with bytes objects.

# Rebuild word symbols as tuple of bytes (each byte as bytes([b]))
def word_to_symtuple(w):
    return tuple(bytes([b]) for b in w.encode("utf-8"))

word_sym = {}
for w, freq in word_freq.items():
    word_sym[w] = (word_to_symtuple(w), freq)

# vocab
vocab = {}
# specials
special_id = 0
special_bytes = {}
for name, tok in special_tokens_cfg.items():
    b = tok.encode("utf-8")
    special_bytes[tok] = b
    vocab[b] = special_id
    special_id += 1
# single bytes
for b in range(256):
    by = bytes([b])
    if by not in vocab:
        vocab[by] = len(vocab)

target_vocab = 2048
merges_list = []  # list of (bytes, bytes)
# Convert word_sym to mutable list of symbols
word_list = [(list(syms), freq) for syms, freq in [(ws, f) for ws, f in [(v[0], v[1]) for v in word_sym.values()]]]
# Actually build properly:
entries = []  # (symbols_list, freq)
for w, (syms, freq) in [(w, (word_sym[w][0], word_sym[w][1])) for w in word_sym]:
    entries.append([list(syms), freq])

def get_pair_counts(entries):
    counts = Counter()
    for syms, freq in entries:
        for i in range(len(syms)-1):
            counts[(syms[i], syms[i+1])] += freq
    return counts

while len(vocab) < target_vocab:
    counts = get_pair_counts(entries)
    if not counts:
        break
    best = max(counts, key=lambda p: counts[p])
    if counts[best] < 2:
        break
    # new symbol
    new_sym = best[0] + best[1]
    if new_sym in vocab:
        # already exists; remove? break
        break
    vocab[new_sym] = len(vocab)
    merges_list.append(best)
    # apply merge to entries
    for e in entries:
        syms = e[0]
        i = 0
        new = []
        while i < len(syms):
            if i < len(syms)-1 and (syms[i], syms[i+1]) == best:
                new.append(new_sym)
                i += 2
            else:
                new.append(syms[i])
                i += 1
        e[0] = new

print("final vocab size", len(vocab))
print("num merges", len(merges_list))

# Save pickles in ncm format:
# vocab: dict bytes->id  (already)
# merges: list of (bytes, bytes) -> ncm builds merges dict {merge: idx}
pickle.dump(vocab, open(os.path.join(out_dir, "bpe_vocab_2048.pickle"), "wb"))
pickle.dump(merges_list, open(os.path.join(out_dir, "bpe_merges_2048.pickle"), "wb"))
print("saved")
PY = None
