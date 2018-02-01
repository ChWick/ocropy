import numpy as np

class Voter:
    def __init__(self, text, distance=0, argnum=-1, filename=None):
        self.text = text
        self.distance = distance

    def __str__(self):
        return "Voter: {%f, %s}" % (self.distance, self.text)

    def compute_distance(self, index, sequences):
        for sequence in sequences:
            diff = np.abs(sequence.count[index] - sequence.median)

            self.distance += diff


class Sequence:
    def __init__(self, key, count, median):
        self.key = key
        self.count = count
        self.median = median

    def __str__(self):
        return "Sequence: {%s, %s, %f}" % (self.key, self.count, self.median)

    def compute_median(self):
        self.median = np.median(self.count)


def add_sequence(sequences, key, reject, index, number_of_voters):
    if key in sequences:
        sequence = sequences[key]
    else:
        sequence = Sequence(key, [0] * number_of_voters, 0 if reject else 1)
        sequences[key] = sequence

    sequence.count[index] += 1


def count_sequences(sequences, index, voters):
    voter = voters[index]
    for start in range(len(voter.text)):
        add_sequence(sequences, voter.text[start:start+2], False, index, len(voters))


def select_voters(voters):
    sequences_dict = {}
    for i, voter in enumerate(voters):
        count_sequences(sequences_dict, i, voters)

    sequences = sequences_dict.values()

    for sequence in sequences:
        sequence.compute_median()

    for i, voter in enumerate(voters):
        voter.compute_distance(i, sequences)

    voters.sort(key=lambda v: v.distance)


def clean_text(text):
    return text.strip()


def text_to_voters(texts):
    return [Voter(clean_text(t)) for t in texts]


def synchronize(texts):
    num_text = len(texts)

    class Sync:
        def __init__(self, substr=None, match=None):
            if substr:
                self.substr = substr
            else:
                self.substr = np.zeros((num_text, 3), dtype=int)

            self.match = match

        def __str__(self):
            return str(self.substr)

        def is_valid(self):
            return np.any(self.substr[:, 2] > 0)

        def lengths(self):
            return self.substr[:, 2]

        def start(self, idx):
            return self.substr[idx, 0]

        def stop(self, idx):
            return self.substr[idx, 1]

        def length(self, idx):
            return self.substr[idx, 2]

        def set_start(self, idx, v):
            self.substr[idx, 0] = v

        def set_stop(self, idx, v):
            self.substr[idx, 1] = v

        def set_length(self, idx, v):
            self.substr[idx, 2] = v

        def set_all(self, idx, v):
            self.substr[idx, :] = v

    # initialize
    def init():
        sync = Sync()
        for i, text in enumerate(texts):
            sync.set_all(i, [0, len(text) - 1, len(text)])

        if sync.is_valid():
            return [sync]

        return []

    def longest_match(maxlen, c1, start1, stop1, c2, start2, stop2):
        mstart1 = 0
        mstart2 = 0
        s1limit = stop1 - maxlen
        s2limit = stop2 - maxlen
        for s1 in range(start1, s1limit + 1):
            for s2 in range(start2, s2limit + 1):
                if c1[s1] == c2[s2]:
                    i1 = s1 + 1
                    i2 = s2 + 1
                    while i1 <= stop1 and i2 <= stop2 and c1[i1] == c2[i2]:
                        i1 += 1
                        i2 += 1

                    increase = i1 - s1 - maxlen
                    if increase > 0:
                        s1limit -= increase
                        s2limit -= increase
                        maxlen += increase
                        mstart1 = s1
                        mstart2 = s2

        return maxlen, mstart1, mstart2

    def save_match(synclist, num_text, sync, start, length, match):
        left, right = Sync(), Sync()
        for i in range(num_text):
            stop = start[i] + length - 1
            left.set_all(i, [sync.start(i), start[i] - 1, start[i] - sync.start(i)])
            right.set_all(i, [stop + 1, sync.stop(i), sync.stop(i) - stop])
            sync.set_all(i, [start[i], stop, length])

        sync.match = match
        if left.is_valid():
            synclist.insert(synclist.index(sync), left)

        if right.is_valid():
            synclist.insert(synclist.index(sync) + 1, right)

    def recursive_sync(synclist, texts, start_index):
        sync = synclist[start_index]
        if np.any(sync.lengths() == 0):
            return

        start = np.zeros(len(texts), dtype=int)
        start[0] = sync.start(0)
        length = sync.length(0)
        for i, text in enumerate(texts[1:], 1):
            length, new_start, start[i] = longest_match(0, texts[0], start[0], start[0] + length - 1,
                                                        text, sync.start(i), sync.stop(i))

            if length == 0:
                return

            change = new_start - start[0]
            if change > 0:
                for j in range(i):
                    start[j] += change

        save_match(synclist, len(texts), sync, start, length, True)

        start_index = synclist.index(sync)
        if start_index - 1 >= 0:
            recursive_sync(synclist, texts, start_index - 1)

        start_index = synclist.index(sync)
        if start_index + 1 < len(synclist):
            recursive_sync(synclist, texts, start_index + 1)

        return

    synclist = init()

    if len(synclist) > 0:
        recursive_sync(synclist, texts, 0)

    return synclist


def perform_vote(inputs, synclist, voters):
    num_candidates = 0
    candidates = [{"char": None, "num_votes": 0} for _ in voters]
    output = ""

    def place_vote(c, num_candidates, num_votes=1):
        index = 0
        if c is not None:
            while index < num_candidates and (candidates[index]['char'] is None or candidates[index]["char"] != c):
                index += 1
        else:
            while index < num_candidates and (candidates[index]['char'] is not None):
                index += 1

        if index < num_candidates:
            candidates[index]['num_votes'] += num_votes
            return num_candidates
        else:
            candidates[i]["char"] = c
            candidates[i]["num_votes"] = num_votes
            return num_candidates + 1

    def winner(num_candidates):
        if num_candidates == 0:
            return True, ""

        leader = 0
        for i in range(1, num_candidates):
            if candidates[i]['num_votes'] > candidates[leader]['num_votes']:
                leader = i

        if candidates[leader]["char"] is None:
            return False, ""

        return True, candidates[leader]["char"]

    for sync in synclist:
        r = True
        while r:
            for i, voter in enumerate(voters):
                if sync.start(i) <= sync.stop(i):
                    num_candidates = place_vote(inputs[i][sync.start(i)], num_candidates)
                    sync.set_start(i, sync.start(i) + 1)
                else:
                    num_candidates = place_vote(None, num_candidates)

            r, out = winner(num_candidates)
            output += out
            num_candidates = 0

    return output


def process_text(texts, optimize=False, n_best=3):
    if len(texts) == 0:
        return texts[0]

    voters = text_to_voters(texts)

    if optimize:
        select_voters(voters)

        if n_best > 0:
            actual_voters = voters[:n_best]
        else:
            actual_voters = voters
    else:
        actual_voters = voters

    inputs = [voter.text for voter in actual_voters]

    synclist = synchronize(inputs)
    return perform_vote(inputs, synclist, actual_voters)


def process_files(files, optimize=False, n_best=3):
    txts = []
    for f in files:
        with open(f, 'r') as content:
            txts.append(content.read())

    return process_text(txts, optimize, n_best)

