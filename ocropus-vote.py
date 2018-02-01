import argparse
from ocrolib.voters.sequence_voter import process_files

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+",
                help="Text files to vote")
parser.add_argument("-O", "--optimize", action="store_true",
                    help="Optimized version")
parser.add_argument("-N", "--n_best", type=int, default=3,
                    help="Number of best voters to use if optimization is activated")

args = parser.parse_args()

print(process_files(args.files, args.optimize, args.n_best))

