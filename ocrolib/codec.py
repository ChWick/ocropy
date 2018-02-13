class Codec:
    """Translate between integer codes and characters."""
    def __init__(self, charset=[], double_as_extra=False):
        self.double_as_extra = double_as_extra
        self.code2char = {}
        self.char2code = {}
        self.init(charset)

    def init(self, charset):
        charset = sorted(list(set(charset)))
        self.code2char = {}
        self.char2code = {}
        for code, char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        return self

    def split(self, sentence):
        if self.double_as_extra:
            r = []
            for i in range(1, len(sentence)):
                if sentence[i] == sentence[i - 1]:
                    r.append(sentence[i] * 2)
                    last_double = True
                else:
                    r.append(sentence[i - 1])
                    last_double = False

            if not last_double:
                r.append(sentence[-1])

            return r
        else:
            return list(sentence)

    def size(self):
        """The total number of codes (use this for the number of output
        classes when training a classifier."""
        return len(list(self.code2char.keys()))

    def encode(self,s):
        "Encode the string `s` into a code sequence."
        # tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in self.split(s)]

    def decode(self,l):
        if self.double_as_extra:
            s = []
            for c in l:
                decoded = self.code2char.get(c, "~")
                if len(s) > 0:
                    if s[-1][-1] == decoded[-1]:
                        if len(s[-1]) == 1:
                            s.pop()
                            s.append(decoded)
                    else:
                        s.append(decoded)
                else:
                    s.append(decoded)
        else:
            "Decode a code sequence into a string."
            s = [self.code2char.get(c,"~") for c in l]
        return s

    def extend(self, codec):
        charset = self.code2char.values()
        size = self.size()
        counter = 0
        for c in codec.code2char.values():
            if not c in charset: # append chars that doesn't appear in the codec
                self.code2char[size] = c
                self.char2code[c] = size
                size += 1
                counter += 1
        print("#", counter, " extra chars added")

    def shrink(self, codec):
        deleted_positions = []
        positions = []
        for number, char in self.code2char.iteritems():
            if not char in codec.char2code and char != "~":
                deleted_positions.append(number)
            else:
                positions.append(number)
        charset = [self.code2char[c] for c in sorted(positions)]
        self.code2char = {}
        self.char2code = {}
        for code, char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        print("#", len(deleted_positions), " unnecessary chars deleted")
        return deleted_positions

ascii_labels = [""," ","~"] + [unichr(x) for x in range(33,126)]

def ascii_codec():
    "Create a codec containing just ASCII characters."
    return Codec().init(ascii_labels)

def ocropus_codec():
    """Create a codec containing ASCII characters plus the default
    character set from ocrolib."""
    import ocrolib
    base = [c for c in ascii_labels]
    base_set = set(base)
    extra = [c for c in ocrolib.chars.default if c not in base_set]
    return Codec().init(base+extra)

