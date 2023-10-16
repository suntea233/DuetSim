import argparse
import sys
import traceback
import logging
from convlab2.evaluator.utterance_diversity import get_diversity_from_dataset
import os
import json
import random
logging.basicConfig(level=logging.DEBUG)


def set_seed(r_seed):
    random.seed(r_seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    default_dataset_path = 'data/multiwoz/train.json'
    parser.add_argument(
        '--dataset_path', required=False,
        help='path to MultiWOZ dataset file (.json), default: {}'.format(
            default_dataset_path),
        default=default_dataset_path, action='store', type=str,
    )
    default_data_key = 'usr'
    parser.add_argument(
        '--data-key', required=False,
        help='whose utterances to get: usr or sys, default: {}'.format(
            default_data_key),
        default=default_data_key, action='store', type=str,
    )
    parser.add_argument(
        '--sample', required=False,
        help='how many conversations to sample before calculating metrics,'
             'default: None', type=int
    )
    args = parser.parse_args()

    try:
        prefix = "diversity" + "_" + args.data_key
        output_file = prefix + "_" + "_".join(args.dataset_path.split("/"))
        # set_seed(20200202)  # 20200202
        diversity = get_diversity_from_dataset(
            dataset_path=args.dataset_path,
            data_key=args.data_key,
            sample=args.sample)

        output_dir = os.path.join('results', 'diversity')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w') as outfile:
            json.dump(diversity, outfile, indent=4)
        print("Done. Saved metrics to {}".format(output_path))

    except Exception as exc:
        logging.error('Cannot process file {}: {} Traceback: {}' .format(
            args.dataset_path, exc, traceback.format_exc()))
        sys.exit()
