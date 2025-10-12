from collections import Counter

class tokenizer:
    def __init__(self, oov_token = "<OOV>", lower = True, num_words = None):
        self.oov_token = oov_token
        self.lower = lower
        self.num_words = num_words
        self.word_index = {oov_token: 1}
        self.index_word = {0: "<PAD>", 1: oov_token}
        self.vocab_size = 2
        self._fitted = False

    def _norm(self, s):
        return s.lower() if self.lower else s

    def fit_on_texts(self, texts):
        counter = Counter()
        for t in texts:
            if isinstance(t, str):
                counter.update(self._norm(t).strip().split())
        items = counter.most_common()
        if self.num_words is not None:
            items = items[: max(0, self.num_words - 2)]
        for tok, _ in items:
            if tok not in self.word_index:
                idx = self.vocab_size
                self.word_index[tok] = idx
                self.index_word[idx] = tok
                self.vocab_size += 1
        self._fitted = True

    def texts_to_sequences(self, texts):
        assert self._fitted, "Tokenizer must be fitted first."
        seqs = []
        for t in texts:
            if not isinstance(t, str):
                seqs.append([])
                continue
            toks = self._norm(t).strip().split()
            seqs.append([self.word_index.get(tok, 1) for tok in toks])
        return seqs

def pad_sequences(sequences, maxlen, padding = "post", truncating = "post", value = 0):
    out = []
    for s in sequences:
        s = list(s)
        if len(s) > maxlen:
            s = s[-maxlen:] if truncating == "pre" else s[:maxlen]
        if len(s) < maxlen:
            pad = [value] * (maxlen - len(s))
            s = pad + s if padding == "pre" else s + pad
        out.append(s)
    return out

def tokenizer_export_state(tok):
    return {
        "config": {
            "oov_token": tok.oov_token,
            "lower": tok.lower,
            "num_words": tok.num_words,
        },
        "word_index": dict(tok.word_index),
        "index_word": {int(k): v for k, v in tok.index_word.items()},
        "vocab_size": int(tok.vocab_size),
        "_fitted": bool(getattr(tok, "_fitted", True)),
    }

def tokenizer_from_state(state):
    tok = tokenizer(
        oov_token=state["config"]["oov_token"],
        lower=state["config"]["lower"],
        num_words=state["config"]["num_words"],
    )
    tok.word_index = dict(state["word_index"])
    tok.index_word = {int(k): v for k, v in state["index_word"].items()}
    tok.vocab_size = int(state["vocab_size"])
    tok._fitted = bool(state.get("_fitted", True))
    return tok