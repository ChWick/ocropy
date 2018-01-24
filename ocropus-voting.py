import argparse
import multiprocessing
import multiprocessing.pool
import numpy as np
import ocrolib
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", nargs="+", type=str, required=True,
                    help="The models to use for voting")
parser.add_argument("-g", "--ground_truth", nargs="+", type=str, required=False,
                    help="If ground truth is provided, the predictions will be evaluated instead of writing them "
                    "to the output dir")
parser.add_argument("-f", "--files", nargs="+", type=str, required=True,
                    help="The files to predict")

parser.add_argument("-o", "--output", type=str, required=False,
                    help="A directory where to output the different models")

parser.add_argument("-l", "--height", default=-1, type=int,
                    help="target line height (overrides recognizer)")
parser.add_argument("-e","--nolineest",action="store_true",
                    help="Disable line estimation")
parser.add_argument("-p","--pad",default=16,type=int,
                    help="extra blank padding to the left and right of text line")
parser.add_argument("-k", "--kind", default="exact",
                    help="kind of comparison (exact, nospace, letdig, letters, digits, lnc), default: %(default)s")

parser.add_argument("--batch_size", default=100, type=int,
                    help="Batch size of the prediction")

args = parser.parse_args()

models = sorted(ocrolib.glob_all(args.models))

if len(models) == 0:
    raise Exception("No models found at %s" % args.models)
else:
    print("Loaded %d models for voting" % len(models))


inputs = sorted(ocrolib.glob_all(args.files))
if len(inputs) == 0:
    raise Exception("No files found at %s" % args.files)
else:
    print("Loaded %d files to predict" % len(inputs))

ground_truth_txts = sorted(ocrolib.glob_all(args.ground_truth))

def load_network(model_path):
    try:
        import ocrolib.tensorflow as tf_backend
        network = tf_backend.SequenceRecognizer.load(model_path)
    except Exception as e:
        print("Ocropus model could not be loaded, trying as normal model: %s" % e)
        network = ocrolib.load_object(model_path, verbose=1)
        for x in network.walk(): x.postLoad()
        for x in network.walk():
            if isinstance(x, ocrolib.lstm.LSTM):
                x.allocate(5000)

    lnorm = getattr(network, "lnorm", None)

    if args.height>0:
        lnorm.setHeight(args.height)

    return {"net": network, "lnorm": lnorm}


def load_line(fname):
    base, _ = ocrolib.allsplitext(fname)
    line = ocrolib.read_image_gray(fname)

    if np.prod(line.shape) == 0:
        return None

    if np.amax(line) == np.amin(line):
        return None

    return line


def prepare_one_for_model(arg):
    fname, lnorm = arg

    line = load_line(fname)

    if not args.nolineest:
        temp = np.amax(line) - line
        temp = temp * 1.0 / np.amax(temp)
        lnorm.measure(temp)
        line = lnorm.normalize(line, cval=np.amax(line))

    line = ocrolib.lstm.prepare_line(line, args.pad)

    return line


def process_model(model):
    print("Starting model %s" % model)
    model = load_network(model)
    network = model["net"]
    lnorm = model["lnorm"]
    load_pool = multiprocessing.Pool(processes=50)
    print("Loading data")
    lines = load_pool.map(prepare_one_for_model, [(fname, lnorm) for fname in inputs])
    print("Predicting data")
    # only one line, atm

    predictions = []
    for i in range(0, len(lines), args.batch_size):
        predictions += [d[:l] for d, l in zip(*network.predict_probabilities(lines[i:i + args.batch_size]))]

    print("Prediction done")

    return predictions, model["net"].codec


pool = multiprocessing.pool.ThreadPool(processes=len(models))
output = pool.map(process_model, models)
#output = list(map(process_model, models))
print("Finished predictions")


def compute_full_codec(codecs):
    full_chars = []
    for codec in codecs:
        full_chars += codec.char2code.keys()

    full_chars = list(set(full_chars))

    return full_chars


all_chars = compute_full_codec([codec for predictions, codec in output])
print(u"Computing full codec as %s" % u"".join(all_chars))

predictions = []
for i, fname in enumerate(inputs):
    predictions.append((fname, [(p[i], codec) for p, codec in output]))


def greedy_decode(pred, code_to_chars, blank=0):
    indices = np.argmax(pred, axis=1)
    decoded = []
    last_i = 0
    for i in indices:
        if i != blank and i != last_i:
            decoded.append(i)
        last_i = i

    return u"".join([code_to_chars[i] for i in decoded])


def process_single_line(fname_preds):
    fname, preds = fname_preds
    print("Processing predictions of %s with %d models" % (fname, len(preds)))
    T = preds[0][0].shape[0]
    logits = np.zeros((T, len(all_chars)), dtype=np.float32)

    misses = 0

    single_predictions = []

    for pred, codec in preds:
        assert(pred.shape[0] == T)
        for all_code, char in enumerate(all_chars):
            if char in codec.char2code:
                model_code = codec.char2code[char]
                logits[:, all_code] += pred[:, model_code]
            else:
                misses += 1

        single_predictions.append(greedy_decode(pred, codec.code2char))

    #logits /= 5

    #do_pred = logits[:, 0] < 0.7

    #logits = logits[do_pred]
    #logits[:, 0] = 0


    #plt.imshow(logits)
    #plt.show()

    return greedy_decode(logits, all_chars), single_predictions


pool = multiprocessing.Pool(processes=8)
txts, single_txts = zip(*pool.map(process_single_line, predictions))
#txts, single_txts = zip(*map(process_single_line, predictions))
#print(txts)

if args.output:
    print("Writing outputs")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i in range(len(models)):
        mdir = os.path.join(args.output, str(i))
        if not os.path.exists(mdir):
            os.mkdir(mdir)

    for input, ts in zip(inputs, single_txts):
        print(input, ts)



if len(ground_truth_txts) > 0:
    print("Evaluating on gt")
    assert(len(ground_truth_txts) == len(txts))

    total_err = 0
    total_chars = 0
    for gt_fname, p in zip(ground_truth_txts, txts):
        gt = ocrolib.project_text(ocrolib.read_text(gt_fname), kind=args.kind)
        txt = ocrolib.project_text(p, kind=args.kind)
        #print(gt, txt)

        err = ocrolib.edist.levenshtein(txt, gt)
        total_err += err
        total_chars += len(gt)

    print(total_err * 1.0 / total_chars)

