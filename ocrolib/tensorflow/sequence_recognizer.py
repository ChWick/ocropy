from .model import Model
from ocrolib.lstm import normalize_nfkc, translate_back, make_target, ctc_align_targets
from scipy.ndimage import measurements,filters
from ocrolib.edist import levenshtein
import numpy as np
import time
from ocrolib import lineest
import matplotlib.pyplot as plt

class SequenceRecognizer:
    @staticmethod
    def load(fname, pretrained=False, codec=None):
        import ocrolib
        data = ocrolib.load_object(fname)
        data["load_file"] = fname
        if codec:
            # overwrite codec with new codec
            data["codec"] = codec
        data["pretrained"] = pretrained
        print(data)
        return SequenceRecognizer(**data)

    """Perform sequence recognition using BIDILSTM and alignment."""
    def __init__(self, Ni, nstates=-1, No=-1, codec=None, normalize=normalize_nfkc, load_file=None, lnorm=None, pretrained=False, model_settings=None):
        self.Ni = Ni
        if codec: No = codec.size()
        self.No = No + 1
        self.debug_align = 0
        self.normalize = normalize
        self.codec = codec
        self.clear_log()
        if lnorm is not None:
            self.lnorm = lnorm
        else:
            self.lnorm = lineest.CenterNormalizer()

        self.model_settings = model_settings

        if load_file is not None:
            if pretrained:
                self.model = Model.create(self.Ni, self.No, self.model_settings)
                self.model.load_weights(load_file)
            else:
                self.model = Model.load(load_file)
        else:
            self.model = Model.create(self.Ni, self.No, self.model_settings)
        self.command_log = []
        self.error_log = []
        self.cerror_log = []
        self.cerror_log_max_size = 1000
        self.error_log_max_size = 1000
        self.key_log = []
        self.last_trial = 0

        self.outputs = []

    def save(self, fname):
        import ocrolib
        data = {"Ni": self.Ni, "No": self.No, "codec": self.codec, "lnorm": self.lnorm, "load_file": fname,
                "normalize": self.normalize, "model_settings": self.model_settings}
        ocrolib.save_object(fname, data)
        self.model.save(fname)

    def walk(self):
        for x in self.lstm.walk(): yield x

    def clear_log(self):
        self.command_log = []
        self.error_log = []
        self.cerror_log = []
        self.key_log = []

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.upgrade()

    def upgrade(self):
        if "last_trial" not in dir(self): self.last_trial = 0
        if "command_log" not in dir(self): self.command_log = []
        if "error_log" not in dir(self): self.error_log = []
        if "cerror_log" not in dir(self): self.cerror_log = []
        if "key_log" not in dir(self): self.key_log = []

    def info(self):
        self.net.info()

    def setLearningRate(self, r, momentum=0.9):
        self.model.default_learning_rate = r

    def predictSequence(self,xs):
        "Predict an integer sequence of codes."
        assert(xs.shape[1]==self.Ni, "wrong image height (image: %d, expected: %d)" % (xs.shape[1], self.Ni))
        # only one batch
        outputs, seq_len, aligned = self.model.decode_sequence([xs])
        aligned = aligned[0]
        return outputs, aligned

    def predict_probabilities(self, xs):
        logits, seq_len = self.model.predict_sequence(xs)
        return logits, seq_len

    def decode_sequences(self, xs):
        logits, seq_len, codes = self.model.decode_sequence(xs)
        return [self.l2s(c) for c in codes]



    def trainSequence(self,xs,cs,update=1,key=None):
        "Train with an integer sequence of codes."
        print(len(xs[0]), len(cs[0]))
        for x in xs: assert(x.shape[-1] == self.Ni, "wrong image height")
        start_time = time.time()
        cost, self.outputs, ler, decoded = self.model.train_sequence(xs, cs)
        print("LSTM-CTC train step took %f s, with ler=%f" % (time.time() - start_time, ler))
        assert(len(xs) == self.outputs.shape[0])
        assert(self.outputs.shape[-1] == self.No)

        # only print first batch entry
        xs = xs[0]
        cs = cs[0]
        self.outputs = self.outputs[0]

        self.aligned = decoded[0]

        self.error = cost
        self.error_log.append(cost)
        # compute class error
        self.cerror = ler
        self.cerror_log.append(ler)

        if len(self.error_log) > self.error_log_max_size:
            del self.error_log[0]

        if len(self.cerror_log) > self.cerror_log_max_size:
            del self.cerror_log[0]

        # training keys
        self.key_log.append(key)

        return decoded[0]

    # we keep track of errors within the object; this even gets
    # saved to give us some idea of the training history
    def errors(self,range=10000,smooth=0):
        result = self.error_log[-range:]
        if smooth>0: result = filters.gaussian_filter(result,smooth,mode='mirror')
        return result

    def cerrors(self,range=10000,smooth=0):
        result = [e*1.0/max(1,n) for e,n in self.cerror_log[-range:]]
        if smooth>0: result = filters.gaussian_filter(result,smooth,mode='mirror')
        return result

    def s2l(self,s):
        "Convert a unicode sequence into a code sequence for training."
        s = self.normalize(s)
        s = [c for c in s]
        return self.codec.encode(s)

    def l2s(self,l):
        "Convert a code sequence into a unicode string after recognition."
        l = self.codec.decode(l)
        return u"".join(l)

    def trainString(self,xs,s,update=1):
        "Perform training with a string. This uses the codec and normalizer."
        return self.trainSequence(xs,self.s2l(s),update=update)

    def predictString(self,xs):
        "Predict output as a string. This uses codec and normalizer."
        return self.decode_sequences([xs])[0]

    def resizeCodec(self, codec):
        print("WARNING: Unsupported codec resizing!")
        assert(self.No == codec.size() + 1)
        return self.codec
