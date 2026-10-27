import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

corpus = "D:/C++/LocalAI/code_corpus.txt"
out_dir = "D:/C++/LocalAI/code_artifacts"
os.makedirs(out_dir, exist_ok=True)

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<|pad|>", "<|eos|>"])
tok.train(files=[corpus], trainer=trainer)
tok.decoder = decoders.ByteLevel()
tok.save(os.path.join(out_dir, "bpe_tokenizer.json"))
print("vocab size", tok.get_vocab_size())
# save vocab/merges pickles for ncm compatibility
import pickle
vocab = tok.get_vocab()
# ncm bpe_tokenizer expects pickles; inspect format
print("sample vocab items", list(vocab.items())[:5])
pickle.dump(vocab, open(os.path.join(out_dir,"bpe_vocab_4096.pickle"),"wb"))
# merges
merges = tok.model.get_merges() if hasattr(tok.model, "get_merges") else []
print("num merges", len(merges))
pickle.dump(merges, open(os.path.join(out_dir,"bpe_merges_4096.pickle"),"wb"))
print("done")
