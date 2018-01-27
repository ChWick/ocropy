import argparse
import multiprocessing
import multiprocessing.pool
import numpy as np
import ocrolib
import os
import matplotlib.pyplot as plt
from ocrolib.prediction_utils import process_model, greedy_decode

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

parser.add_argument("--threads", default=8, type=int,
                    help="The number of threads to use")

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

models = [
    {"path": m,
     "batch_size": args.batch_size,
     "height": args.height,
     "nolineest": args.nolineest,
     "threads": args.threads,
     "pad": args.pad,
     "predict": "probabilities",
     }
    for m in models
]

pool = multiprocessing.pool.ThreadPool(processes=min(len(models), args.threads))
output = pool.map(process_model, [(model, inputs) for model in models])
#output = list(map(process_model, [(m, inputs) for m in models]))
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
                # if model_code == 0:
                logits[:, all_code] += pred[:, model_code]
                # else:
                #     sp = pred[:, model_code]
                #     op = sp[:]
                #     for i in range(len(sp)):
                #         op[i] = max(sp[max(0, i - 1):min(i+1,len(sp))])
                #
                #     logits[:, all_code] = op
            else:
                misses += 1

        single_predictions.append(greedy_decode(pred, codec.code2char))

    logits /= len(models)

    # do_pred = logits[:, 0] < 0.9

    # logits = logits[do_pred]
    # logits[:, 0] = 0


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

    for fname, ts in zip(inputs, single_txts):
        fname = os.path.basename(fname)
        base,_ = ocrolib.allsplitext(fname)
        for i, txt in enumerate(ts):
            mdir = os.path.join(args.output, str(i))
            ocrolib.write_text(os.path.join(mdir, base+".txt"), txt)

    print("Outputs written to %s" % args.output)



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

