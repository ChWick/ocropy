import numpy as np

from ocrolib.prediction_utils import greedy_decode


def decode_probs(preds, codecs, ctc_merge_repeated=True):
    def compute_full_codec(codecs):
        full_chars = []
        for codec in codecs:
            full_chars += codec.char2code.keys()

        full_chars = list(set(full_chars))

        return full_chars

    all_chars = compute_full_codec(codecs)
    T = preds[0].shape[0]
    logits = np.zeros((T, len(all_chars)), dtype=np.float32)

    misses = 0

    for pred, codec in zip(preds, codecs):
        assert(pred.shape[0] == T)
        for all_code, char in enumerate(all_chars):
            if char in codec.char2code:
                model_code = codec.char2code[char]
                logits[:, all_code] += pred[:, model_code]
            else:
                misses += 1

    logits /= len(preds)

    return greedy_decode(logits, all_chars, ctc_merge_repeated=ctc_merge_repeated)

