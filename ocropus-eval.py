import argparse
import multiprocessing
import multiprocessing.pool
import numpy as np
import ocrolib
import os
import matplotlib.pyplot as plt
from ocrolib.prediction_utils import process_model, greedy_decode, load_gt
from collections import Counter
import pickle

parser = argparse.ArgumentParser()

# global setup
parser.add_argument("--threads", default=1, type=int,
                    help="The number of threads in the global threads pool!")
parser.add_argument("--max_parallel_models", default=10, type=int,
                    help="The maximum amount of models that are run in parallel")
# evaluation setup
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

args = parser.parse_args()
if args.output:
    args.output = os.path.expanduser(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

models = sorted(ocrolib.glob_all(args.models))

if len(models) == 0:
    raise Exception("No models found at %s" % args.models)
else:
    print("Loaded %d models for evaluating" % len(models))

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
     "global_threads": args.threads,
     "single_threads": max(1, args.threads // len(models)),
     "pad": args.pad,
     "predict": "decode_probabilities",
     }
    for m in models
]

pool = multiprocessing.pool.ThreadPool(processes=min(args.max_parallel_models, len(models)))
output = pool.map(process_model, [(model, inputs) for model in models])
#pickle.dump(output, open(os.path.expanduser("~/test.dump"), 'w'))
#output = pickle.load(open(os.path.expanduser("~/test.dump"), 'r'))
codecs = [c for _, c in output]


def evaluate_single(params):
    print("Evaluating on gt")
    prediction, codec = params

    assert(len(prediction) == len(raw_gt))

    total_err = 0
    total_chars = 0
    counts = Counter()
    for gt, (decodec, probs) in zip(raw_gt, prediction):
        txt = ocrolib.project_text(decodec, kind=args.kind)

        err, cs = ocrolib.edist.xlevenshtein(txt, gt, context=args.context)

        total_err += err
        total_chars += len(gt)
        if args.confusion > 0:
            for u, v in cs:
                counts[(u, v)] += 1

    return total_err * 1.0 / total_chars, counts, total_err, total_chars


pool = multiprocessing.Pool(processes=min(len(models), args.threads))
scores = pool.map(evaluate_single, output)
pool.close()


def output_single(params):
    print("Output of single model")
    (prediction, codec), model = params
    for input, (decoded, probs) in zip(inputs, prediction):
        base, _ = ocrolib.allsplitext(os.path.split(input)[1])
        if len(models) > 1:
            m, _ = ocrolib.allsplitext(os.path.split(model["path"])[1])
            filename = os.path.join(args.output, m, base + ".txt")
        else:
            filename = os.path.join(args.output, base + ".txt")

        ocrolib.write_text(filename, decoded)


if args.output:
    print("Output to %s" % args.output)
    pool = multiprocessing.Pool(processes=min(len(models), args.threads))
    pool.map(output_single, [(o, m) for o, m in zip(output, models)])
    pool.close()


def compute_voting(output, voter_type):
    import ocrolib.voters.sequence_voter as sequence_voter
    import ocrolib.voters.probability_voter as probability_voter

    print("Running voting")
    if len(output) == 0:
        raise Exception("No models evaluted!")

    total_err = 0
    total_chars = 0
    counts = Counter()
    num_examples = len(output[0][0])

    for example in range(num_examples):
        gt = raw_gt[example]

        model_predictions = []
        model_probs = []
        for model in range(len(output)):
            decoded, probs = output[model][0][example]
            txt = ocrolib.project_text(decoded, kind=args.kind)
            model_predictions.append(txt)
            model_probs.append(probs)

        if voter_type == "sequence":
            voted = sequence_voter.process_text(model_predictions, True, -1)
        elif voter_type == "prob":
            voted = probability_voter.decode_probs(model_probs, codecs, ctc_merge_repeated=True)

        err, cs = ocrolib.edist.xlevenshtein(voted, gt, context=args.context)

        total_err += err
        total_chars += len(gt)
        if args.confusion > 0:
            for u, v in cs:
                counts[(u, v)] += 1

    return total_err * 1.0 / total_chars, counts



print("Index\tName\tScore\tTotal Errs\tTotal Chars")
for i, (model, (score, counts, total_errs, total_chars)) in enumerate(zip(models, scores)):
    print("%03d\t%s\t%f\t%d\t%d" % (i, model['path'], score, total_errs, total_chars))


print("Confusion matrices")
for i, (model, (score, counts, total_errs, total_chars)) in enumerate(zip(models, scores)):
    print("%03d\t%s\t%f" % (i, model['path'], score))
    if args.confusion > 0:
        for (a, b), v in counts.most_common(args.confusion):
            print("%d\t%s\t%s" % (v, a.encode("utf-8"), b.encode("utf-8")))


if len(models) > 1:
    print("Voting %d models" % len(models))
    def run_vote(voter):
        voted_err, voted_counts = compute_voting(output, voter)
        print("%s\t%s\t%f" % (voter, "Voted", voted_err))
        if args.confusion > 0:
            for (a, b), v in voted_counts.most_common(args.confusion):
                print("%d\t%s\t%s" % (v, a.encode("utf-8"), b.encode("utf-8")))

    for voter in ["sequence", "prob"]:
        run_vote(voter)
else:
    print("No voting required")



