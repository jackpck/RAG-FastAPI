def mask_block(text: str,
               mask_prob: float,
               block_size: int = 3,
               random_seed: int = 123) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    sentence_block = [" ".join(sentences[i: i+block_size]) for i in range(0, len(sentences), block_size)]

    rng = np.random.default_rng(seed=random_seed)
    sentence_block_masked = []
    masked_block = []
    for sentence in sentence_block:
        if (rng.random() < mask_prob) and (sentence_block_masked[-1] != "[MASK]"):
            sentence_block_masked.append("[MASK]")
            masked_block.append(sentence)
        else:
            sentence_block_masked.append(sentence)
    return sentence_block_masked, masked_block