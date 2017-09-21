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
    def load(fname):
        import ocrolib
        data = ocrolib.load_object(fname)
        data["load_file"] = fname
        print(data)
        return SequenceRecognizer(**data)

    """Perform sequence recognition using BIDILSTM and alignment."""
    def __init__(self, Ni, nstates=-1, No=-1, codec=None, normalize=normalize_nfkc, load_file=None, lnorm=None):
        self.Ni = Ni
        if codec: No = codec.size()
        self.No = No + 1
        self.learning_rate = 1e-2
        self.debug_align = 0
        self.normalize = normalize
        self.codec = codec
        self.clear_log()
        if lnorm is not None:
            self.lnorm = lnorm
        else:
            self.lnorm = lineest.CenterNormalizer()

        if load_file is not None:
            self.model = Model.load(load_file, learning_rate=self.learning_rate)
        else:
            self.model = Model.create(self.Ni, nstates, self.No, learning_rate=self.learning_rate)
        self.command_log = []
        self.error_log = []
        self.cerror_log = []
        self.key_log = []
        self.last_trial = 0

        self.outputs = []

    def save(self, fname):
        import ocrolib
        data = {"Ni": self.Ni, "No": self.No, "codec": self.codec, "lnorm": self.lnorm, "load_file": fname,
                "normalize": self.normalize}
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
        self.model.learning_rate = r

    def predictSequence(self,xs):
        "Predict an integer sequence of codes."
        assert(xs.shape[1]==self.Ni, "wrong image height (image: %d, expected: %d)"%(xs.shape[1],self.Ni))
        # only one batch
        self.outputs = self.model.predict_sequence([xs])[0]
        return translate_back(self.outputs)

    def trainSequence(self,xs,cs,update=1,key=None):
        "Train with an integer sequence of codes."
        for x in xs: assert(x.shape[-1] == self.Ni, "wrong image height")
        start_time = time.time()
        cost, self.outputs = self.model.train_sequence(xs, cs)
        print("LSTM-CTC train step took %f s" % (time.time() - start_time))
        assert(len(xs) == self.outputs.shape[0])
        assert(self.outputs.shape[-1] == self.No)

        # only print first batch entry
        xs = xs[0]
        cs = cs[0]
        self.outputs = self.outputs[0]

        result = translate_back(self.outputs)

        self.targets = np.array(make_target(cs, self.No))
        self.aligned = np.array(ctc_align_targets(self.outputs,self.targets,debug=self.debug_align))

        self.error = np.sum(cost ** 2)
        self.error_log.append(self.error ** .5 / len(cs))
        # compute class error
        self.cerror = levenshtein(cs, result)
        self.cerror_log.append((self.cerror, len(cs)))

        # training keys
        self.key_log.append(key)

        return result

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
        cs = self.predictSequence(xs)
        return self.l2s(cs)
