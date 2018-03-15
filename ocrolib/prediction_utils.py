import multiprocessing
import ocrolib
import numpy as np
from click import progressbar

def load_network(model):
    import ocrolib.tfmodel as tf_backend
    network = tf_backend.SequenceRecognizer.load(model["path"], threads=model["global_threads"])

    lnorm = getattr(network, "lnorm", None)

    if model["height"] > 0:
        lnorm.setHeight(model["height"])

    model["lnorm"] = lnorm

    return network, model


def load_line(fname):
    base, _ = ocrolib.allsplitext(fname)
    line = ocrolib.read_image_gray(fname)

    if np.prod(line.shape) == 0:
        return None

    if np.amax(line) == np.amin(line):
        return None

    return line


def prepare_one_for_model(arg):
    fname, model = arg
    lnorm = model["lnorm"]
    nolineest = model["nolineest"]
    pad = model["pad"]

    line = load_line(fname)

    if not nolineest:
        temp = np.amax(line) - line
        temp = temp * 1.0 / np.amax(temp)
        lnorm.measure(temp)
        line = lnorm.normalize(line, cval=np.amax(line))

    line = ocrolib.lstm.prepare_line(line, pad)

    return line


def process_model(args):
    model, inputs = args
    print("Starting model %s" % model["path"])
    network, model = load_network(model)
    load_pool = multiprocessing.Pool(processes=model["single_threads"])
    print("Loading data")
    lines = load_pool.map(prepare_one_for_model, [(fname, model) for fname in inputs])
    load_pool.close()
    # lines = list(map(prepare_one_for_model, [(fname, model) for fname in inputs]))
    print("Predicting data")

    predictions = []
    if model["predict"] == "probabilities":
        with progressbar(range(0, len(lines), model["batch_size"])) as start_indices:
            for i in start_indices:
                predictions += [d[:l] for d, l in zip(*network.predict_probabilities(lines[i:i + model["batch_size"]]))]
    elif model["predict"] == "decode":
        with progressbar(range(0, len(lines), model["batch_size"])) as start_indices:
            for i in start_indices:
                predictions += network.decode_sequences(lines[i:i + model["batch_size"]])
    elif model["predict"] == "decode_probabilities":
        with progressbar(range(0, len(lines), model["batch_size"])) as start_indices:
            for i in start_indices:
                predictions += network.decode_sequences(lines[i:i + model["batch_size"]], decoded_only=False)
    else:
        raise Exception("Unknown prediction requested: '%s'" % model["predict"])


    print("Prediction done")

    return predictions, network.codec


def greedy_decode(pred, code_to_chars, blank=0, ctc_merge_repeated=True):
    indices = np.argmax(pred, axis=1)
    decoded = []
    last_i = 0
    for i in indices:
        if i != blank and not (ctc_merge_repeated and i == last_i):
            decoded.append(i)
        last_i = i

    return u"".join([code_to_chars[i] for i in decoded])


def load_gt(gt_files, kind):
    return [ocrolib.project_text(ocrolib.read_text(f), kind=kind) for f in gt_files]
