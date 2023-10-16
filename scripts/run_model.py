import torch
import argparse
import sys
import random
import numpy as np
import traceback
import logging
from convlab2.nlg.generative_models import MODEL_ID_MODEL_CLASS_MAPPING

logging.basicConfig(level=logging.DEBUG)


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    default_model_id = 'google/flan-t5-small'
    parser.add_argument(
        '--model-id', required=False,
        choices=MODEL_ID_MODEL_CLASS_MAPPING.keys(),
        help='default: {}'.format(default_model_id),
        default=default_model_id, action='store', type=str,
    )
    default_prompt_filepath = 'tests/sample_prompt.txt'
    parser.add_argument(
        '--prompt-filepath', required=False,
        help='default: {}'.format(default_prompt_filepath),
        default=default_prompt_filepath, action='store', type=str,
    )
    default_temparature = 0.75
    parser.add_argument(
        '--temperature', required=False,
        help='default: {}'.format(default_temparature),
        default=default_temparature, type=float
    )
    default_tolerance = 20
    parser.add_argument(
        '--tolerance', required=False,
        help='default: {}'.format(default_tolerance),
        default=default_tolerance, type=int
    )
    default_process_model_output = 'FALSE'
    parser.add_argument(
        '--process-model-output', required=False,
        choices={'TRUE', 'FALSE'},
        default=default_process_model_output, type=str,
        help='default: {}'.format(default_process_model_output)
    )
    default_random_seed = 0
    parser.add_argument(
        '--random-seed', required=False,
        default=default_random_seed, type=int,
        help='default: {}'.format(default_random_seed)
    )
    default_debug_prompt_processing = 'FALSE'
    parser.add_argument(
        '--debug-prompt-processing', required=False,
        choices={'TRUE', 'FALSE'},
        default=default_debug_prompt_processing, type=str,
        help='default: {}'.format(default_debug_prompt_processing)
    )
    args = parser.parse_args()
    debug_prompt_processing = False if args.debug_prompt_processing == 'FALSE' else True  # noqa
    process_model_output = False if args.process_model_output == 'FALSE' else True  # noqa

    set_seed(args.random_seed)

    try:
        prompt_text = open(args.prompt_filepath, 'r').read()
    except Exception as exc:
        logging.error('Cannot read file {}: {} Traceback: {}' .format(
            args.prompt_filepath, exc, traceback.format_exc()))
        sys.exit()

    try:
        model_class = MODEL_ID_MODEL_CLASS_MAPPING.get(args.model_id)
        model = model_class(args.model_id)
    except Exception as exc:
        logging.error('Cannot load model {}: {} Traceback: {}' .format(
            args.model_id, exc, traceback.format_exc()))
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if process_model_output:
        generated_text = model.generate(
            prompt_text=prompt_text, temperature=args.temperature,
            tolerance=args.tolerance,
            debug_prompt_processing=debug_prompt_processing
        )
    else:
        generated_text = model._generate_text(
            prompt_text=prompt_text, temperature=args.temperature,
            debug_prompt_processing=debug_prompt_processing
        )

    print('GENERATED TEXT: \n\n{}'.format(generated_text))
