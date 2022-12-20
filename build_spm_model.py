## Utility to train a sentence piece model from the given caption_corpus_val2017.txt
# SentencePiece software employs the BPE algorithm, but it applies it on Unicode(reference_text) rather than directly applying it on reference text.

import sentencepiece as spm

spm.SentencePieceTrainer.train(input='caption_corpus_val2017.txt', model_prefix='spm1', vocab_size=10000, character_coverage=1.0, model_type='bpe')
