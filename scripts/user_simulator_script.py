import argparse
# available NLU models
# from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.e2e.user_simulator.user_simulator import UserSimulatorE2E
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
# from convlab2.nlu.milu.multiwoz import MILU
# available DST models
from convlab2.dst.rule.multiwoz import RuleDST
# from convlab2.dst.mdbt.multiwoz import MDBT
# from convlab2.dst.sumbt.multiwoz import SUMBT
# from convlab2.dst.trade.multiwoz import TRADE
# from convlab2.dst.comer.multiwoz import COMER
# available Policy models
from convlab2.policy.rule.multiwoz import RulePolicy
# from convlab2.policy.ppo.multiwoz import PPOPolicy
# from convlab2.policy.pg.multiwoz import PGPolicy
# from convlab2.policy.mle.multiwoz import MLEPolicy
# from convlab2.policy.gdpl.multiwoz import GDPLPolicy
# from convlab2.policy.vhus.multiwoz import UserPolicyVHUS
# from convlab2.policy.mdrg.multiwoz import MDRGWordPolicy
# from convlab2.policy.hdsa.multiwoz import HDSA
# from convlab2.policy.larl.multiwoz import LaRL
# available NLG models
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlg.sclstm.multiwoz import SCLSTM
from convlab2.nlg.scgpt.multiwoz import SCGPT
# available E2E models
# from convlab2.e2e.sequicity.multiwoz import Sequicity
# from convlab2.e2e.damd.multiwoz import Damd
from convlab2.dialog_agent import PipelineAgent
# from convlab2.dialog_agent import BiSession
# from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer

from convlab2.nlg.generative_models import MODEL_ID_MODEL_CLASS_MAPPING

import random
import numpy as np
import torch
import os


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    default_model_id = 'azure_openai/gpt-35-turbo'
    parser.add_argument(
        '--model-id', required=False,
        choices=MODEL_ID_MODEL_CLASS_MAPPING.keys(),
        default=default_model_id, action='store', type=str,
        help='default: {}'.format(default_model_id)
    )
    default_print_details = 'TRUE'
    parser.add_argument(
        '--print-details', required=False,
        choices={'TRUE', 'FALSE'},
        default=default_print_details, type=str,
        help='default: {}'.format(default_print_details)
    )
    default_debug_prompt_processing = 'FALSE'
    parser.add_argument(
        '--debug-prompt-processing', required=False,
        choices={'TRUE', 'FALSE'},
        default=default_debug_prompt_processing, type=str,
        help='default: {}'.format(default_debug_prompt_processing)
    )
    default_total_dialog = 100
    parser.add_argument(
        '--total-dialog', required=False,
        help='default: {}'.format(default_total_dialog),
        default=default_total_dialog, type=int
    )

    default_task_description_file_path = r'C:\Users\administr\Desktop\AAAI\prompts\task_description_5.txt'
    parser.add_argument(
        '--task-description-file-path', required=False,
        help='default: {}'.format(default_task_description_file_path),
        default=default_task_description_file_path, type=str
    )
    default_save_dir = 'results'
    parser.add_argument(
        '--save-dir', required=False,
        help='default: {}'.format(default_save_dir),
        default=default_save_dir, type=str
    )
    default_jaccard_sampling = 'TRUE'
    parser.add_argument(
        '--jaccard-sampling', required=False,
        choices={'TRUE', 'FALSE'},
        default=default_jaccard_sampling, type=str,
        help='use Jaccard sampling, if False: use random sampling, '
             'default: {}'.format(default_jaccard_sampling)
    )
    sys_nlu_default = True
    parser.add_argument(
        '--sys-nlu', required=False,
        choices={'TRUE', 'FALSE'},
        default=sys_nlu_default, type=str,
        help='default: {}'.format(sys_nlu_default)
    )

    default_random_seed = 20200202
    parser.add_argument(
        '--random-seed', required=False,
        default=default_random_seed, type=int,
        help='default: {}'.format(default_random_seed)
    )

    default_user_simulator = 'rulebased'
    user_simulator_values = [default_user_simulator, 'rulebased']
    parser.add_argument(
        '--user-simulator', required=False,
        default=default_user_simulator, type=str,
        choices=user_simulator_values,
        help='default: {}'.format(default_user_simulator)
    )

    default_user_nlg = 'scgpt'
    user_nlg_values = [default_user_nlg, 'sclstm','scgpt']
    parser.add_argument(
        '--user-nlg', required=False,
        default=default_user_nlg, type=str,
        choices=user_nlg_values,
        help='default: {}'.format(default_user_nlg)
    )

    default_temperature = 0.8
    parser.add_argument(
        '--temperature', required=False,
        default=default_temperature, type=float,
        help='default: {}'.format(default_temperature)
    )

    default_num_shots = 2
    parser.add_argument(
        '--num-shots', required=False,
        default=default_num_shots, type=int,
        help='default: {}'.format(default_num_shots)
    )

    default_use_bullet_pointed_goal = 'FALSE'
    parser.add_argument(
        '--use-bullet-pointed-goal', required=False,
        choices={'TRUE', 'FALSE'},
        default=default_use_bullet_pointed_goal, type=str,
        help='default: {}'.format(default_use_bullet_pointed_goal)
    )

    args = parser.parse_args()
    print_details = False if args.print_details == 'FALSE' else True
    jaccard_sampling = (
        False if args.jaccard_sampling == 'FALSE' else True
    )
    debug_prompt_processing = (
        False if args.debug_prompt_processing == 'FALSE' else True
    )
    use_bullet_pointed_goal = (
        False if args.use_bullet_pointed_goal == 'FALSE' else True
    )
    sys_nlu_user_sim = (
        False if args.sys_nlu == 'FALSE' else True
    )

    save_dir = os.path.join(args.save_dir, args.user_simulator)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # go to README.md of each model for more information
    # BERT nlu
    sys_nlu = BERTNLU(
        mode='all',
        config_file='multiwoz_all_context.json',
        model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_all_context.zip')  # noqa

    # sys_nlu = SVMNLU(mode='sys')
    # simple rule DST
    sys_dst = RuleDST()
    # rule policy
    sys_policy = RulePolicy()
    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    # assemble
    sys_agent = PipelineAgent(
        sys_nlu, sys_dst, sys_policy, sys_nlg,
        name='sys',
        print_details=print_details)

    if args.user_simulator == 'ours':
        user_nlg_model_class = MODEL_ID_MODEL_CLASS_MAPPING.get(
            args.model_id)
        user_nlg = user_nlg_model_class(args.model_id)
        user_agent = (
            UserSimulatorE2E(
                sample_based_on_user_goal=jaccard_sampling,
                shots_file=r'C:\Users\administr\Desktop\AAAI\data\multiwoz\train_modified.json',
                nlg=user_nlg, print_details=print_details,
                task_description_file_path=args.task_description_file_path,
                debug_prompt_processing=debug_prompt_processing,
                use_bullet_pointed_goal=use_bullet_pointed_goal,
                num_shots=args.num_shots,
                sys_nlu=sys_nlu_user_sim,
                temperature=args.temperature
            )
        )
    elif args.user_simulator == 'rulebased':  # baseline

        # BERT nlu trained on sys utterance
        # user_nlu = SVMNLU(mode='usr')
        user_nlu = BERTNLU(
            mode='sys', config_file='multiwoz_sys_context.json',
            model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_sys_context.zip')  # noqa

        # not use dst
        user_dst = None
        # rule policy
        user_policy = RulePolicy(character='usr')  #
        if args.user_nlg == 'sclstm':
            print('user_nlg={}'.format(args.user_nlg))
            user_nlg = SCLSTM(is_user=True)
        elif args.user_nlg == "scgpt":
            print('user_nlg={}'.format(args.user_nlg))
            user_nlg = SCGPT(is_user=True)
        else:
            # template NLG
            user_nlg = TemplateNLG(is_user=True)

        # assemble
        user_agent = PipelineAgent(
            user_nlu, user_dst, user_policy, user_nlg, name='user')
    else:
        print('user_simulator={} not defined'.format(args.user_simulator))
        exit()

    analyzer = Analyzer(
        user_agent=user_agent, dataset='multiwoz', save_dir=save_dir)
    print(args)
    set_seed(args.random_seed)
    analyzer.comprehensive_analyze(
        sys_agent=sys_agent, model_name=args.user_simulator,
        total_dialog=args.total_dialog
    )
