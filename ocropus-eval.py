import argparse
import multiprocessing
import multiprocessing.pool
import numpy as np
import ocrolib
import os
import matplotlib.pyplot as plt
from ocrolib.prediction_utils import process_model, greedy_decode, load_gt
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", nargs="+", type=str, required=True,
                    help="The models to evaluate")
parser.add_argument("-g", "--ground_truth", nargs="+", type=str, required=False,
                    help="If ground truth is provided, the predictions will be evaluated instead of writing them "
                         "to the output dir")
parser.add_argument("-f", "--files", nargs="+", type=str, required=True,
                    help="The files to predict")

parser.add_argument("-o", "--output", type=str, required=False,
                    help="A directory where to output the prediction")

parser.add_argument("-l", "--height", default=-1, type=int,
                    help="target line height (overrides recognizer)")
parser.add_argument("-e","--nolineest",action="store_true",
                    help="Disable line estimation")
parser.add_argument("-p","--pad",default=16,type=int,
                    help="extra blank padding to the left and right of text line")
parser.add_argument("-k", "--kind", default="exact",
                    help="kind of comparison (exact, nospace, letdig, letters, digits, lnc), default: %(default)s")
parser.add_argument("-C", "--context",
                    type=int, default=0, help="context for confusion matrix, default: %(default)s")
parser.add_argument("-c", "--confusion", default=10, type=int,
                    help="output this many top confusion, default: %(default)s")

parser.add_argument("--batch_size", default=100, type=int,
                    help="Batch size of the prediction")
parser.add_argument("--threads", default=8, type=int,
                    help="The number of threads to use")
parser.add_argument("--load_threads", default=20, type=int,
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
print("Loaded %d files as ground truth" % len(ground_truth_txts))
if len(ground_truth_txts) != len(inputs):
    print("Warning: Mismatch in number of inputs and ground truth")

    # create set of
    gt_base_paths = [os.path.join(os.path.dirname(o), os.path.basename(o)[:os.path.basename(o).find(".")]) for o in ground_truth_txts]
    inputs_base_paths = [os.path.join(os.path.dirname(o), os.path.basename(o)[:os.path.basename(o).find(".")]) for o in inputs]

    valid_paths = set(gt_base_paths).intersection(set(inputs_base_paths))

    ground_truth_txts = [gt + ".gt.txt" for gt in valid_paths]
    valid_inputs = []
    for valid_path in valid_paths:
        for input in inputs:
            if input.startswith(valid_path):
                valid_inputs.append(input)
                break

    inputs = valid_inputs
    print("Valid files: %d" % len(inputs))

    assert(len(ground_truth_txts) == len(inputs))

raw_gt = load_gt(ground_truth_txts, args.kind)


models = [
    {"path": m,
     "batch_size": args.batch_size,
     "height": args.height,
     "nolineest": args.nolineest,
     "threads": args.threads,
     "load_threads": args.load_threads if args.load_threads > 0 else args.theads,
     "pad": args.pad,
     "predict": "decode",
     }
    for m in models
]

pool = multiprocessing.pool.ThreadPool(processes=min(len(models), args.threads))
output = pool.map(process_model, [(model, inputs) for model in models])


def evaluate_single(params):
    print("Evaluating on gt")
    prediction, codec = params

    assert(len(prediction) == len(raw_gt))

    total_err = 0
    total_chars = 0
    counts = Counter()
    for gt, p in zip(raw_gt, prediction):
        txt = ocrolib.project_text(p, kind=args.kind)

        err, cs = ocrolib.edist.xlevenshtein(txt, gt, context=args.context)

        total_err += err
        total_chars += len(gt)
        if args.confusion > 0:
            for u, v in cs:
                counts[(u, v)] += 1

    return total_err * 1.0 / total_chars, counts


pool = multiprocessing.Pool(processes=min(len(models), args.threads))
scores = pool.map(evaluate_single, output)

print("Index\tName\tScore")
for i, (model, (score, counts)) in enumerate(zip(models, scores)):
    print("%03d\t%s\t%f" % (i, model['path'], score))

print("Confusion matrices")
for i, (model, (score, counts)) in enumerate(zip(models, scores)):
    print("%03d\t%s\t%f" % (i, model['path'], score))
    if args.confusion > 0:
        for (a, b), v in counts.most_common(args.confusion):
            print("%d\t%s\t%s" % (v, a.encode("utf-8"), b.encode("utf-8")))



