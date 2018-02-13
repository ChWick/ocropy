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
import time

parser = argparse.ArgumentParser()

# global setup
parser.add_argument("--threads", default=1, type=int,
                    help="The number of threads in the global threads pool!")
parser.add_argument("--max_parallel_models", default=10, type=int,
                    help="The maximum amount of models that are run in parallel")
parser.add_argument("--timing", default=False, action="store_true",
                    help="Output the total time required for evaluation.")
parser.add_argument("--output", default=False, action="store_true",
                    help="Write the final prediction to the data dir")
# evaluation setup
parser.add_argument("-m", "--models", nargs="+", type=str, required=True,
                    help="The models to evaluate")
parser.add_argument("-f", "--files", nargs="+", type=str, required=True,
                    help="The files to predict")

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

total_start_time = time.time()

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


models = [
    {"path": m,
     "batch_size": args.batch_size,
     "height": args.height,
     "nolineest": args.nolineest,
     "global_threads": args.threads,
     "single_threads": max(1, args.threads // min(args.max_parallel_models, len(models))),
     "pad": args.pad,
     "predict": "decode_probabilities",
     }
    for m in models
]

if args.max_parallel_models == 1:
    output = list(map(process_model, [(model, inputs) for model in models]))
else:
    pool = multiprocessing.pool.ThreadPool(processes=min(args.max_parallel_models, len(models)))
    output = pool.map(process_model, [(model, inputs) for model in models])
    pool.close()

codecs = [c for _, c in output]



def compute_voting(output, voter_type):
    import ocrolib.voters.sequence_voter as sequence_voter
    import ocrolib.voters.probability_voter as probability_voter

    print("Running voting")
    if len(output) == 0:
        raise Exception("No models evaluted!")

    num_examples = len(output[0][0])

    voted_sequences = []
    for example in range(num_examples):
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

        else:
            raise Exception("Unknown voter %s" % voter_type)

        voted_sequences.append(voted)

    return voted_sequences


if len(models) > 1:
    print("Voting %d models" % len(models))
    predicted_sentences = compute_voting(output, "sequence")
else:
    print("No voting required")
    predicted_sentences = [decoded for decoded, _ in output[0][0]]

print(predicted_sentences)
if args.timing:
    print("Total time required %f s for %d lines" % (time.time() - total_start_time, len(inputs)))


def output_single(prediction):
    assert(len(prediction) == len(inputs))
    for input, pred in zip(inputs, prediction):
        base, _ = ocrolib.allsplitext(input)
        ocrolib.write_text(base+".txt", pred)


if args.output:
    base, _ = os.path.split(inputs[0])
    print("Output to %s" % base)
    output_single(predicted_sentences)

