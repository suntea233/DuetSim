import argparse
# available NLU models
# from convlab2.nlu.svm.multiwoz import SVMNLU
from user_simulator import LLM_US
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
# from convlab2.nlu.milu.multiwoz import MILU
# available DST models
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.nlu.svm.multiwoz import SVMNLU
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
# available E2E models
# from convlab2.e2e.sequicity.multiwoz import Sequicity
# from convlab2.e2e.damd.multiwoz import Damd
from convlab2.dialog_agent import PipelineAgent
# from convlab2.dialog_agent import BiSession
# from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer


import random
import numpy as np
import torch
import os
os.environ['SAFETENSORS_FAST_GPU']='1'
os.environ['PYDEVD_USE_CYTHON']='NO'



def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)



def run():
    # go to README.md of each model for more information
    # BERT nlu
    # sys_nlu = BERTNLU(
    #     mode='all',
    #     config_file='multiwoz_all_context.json',
    #     model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_all_context.zip')  # noqa
    sys_nlu = None
    # sys_nlu = SVMNLU(mode='sys')
    # simple rule DST
    sys_dst = RuleDST()
    # sys_dst = None
    # rule policy
    sys_policy = RulePolicy()
    # template NLG
    # sys_nlg = TemplateNLG(is_user=False)
    sys_nlg = None
    # assemble

    print_details = True
    sys_agent = PipelineAgent(
        sys_nlu, sys_dst, sys_policy, sys_nlg,
        name='sys',
        print_details=print_details)

    # user_nlu = None
    # user_dst = None
    # user_policy = RulePolicy(character='usr')
    # user_nlg = TemplateNLG()
    # user_agent = PipelineAgent(
    #     user_nlu,user_dst,user_policy,user_nlg,
    #     name='user',
    #     print_details=print_details
    # )

    user_agent = LLM_US(model_name='flant5')

    # user_nlu = BERTNLU(
    #     mode='sys', config_file='multiwoz_sys_context.json',
    #     model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_sys_context.zip')  # noqa
    #
    # # not use dst
    # user_dst = None
    # # rule policy
    # user_nlg = "template"
    # user_policy = RulePolicy(character='usr')  #
    # if user_nlg == 'sclstm':
    #     print('user_nlg={}'.format(user_nlg))
    #     user_nlg = SCLSTM(is_user=True)
    # elif user_nlg == "scgpt":
    #     print('user_nlg={}'.format(user_nlg))
    #     user_nlg = SCGPT(is_user=True)
    # else:
    #     # template NLG
    #     user_nlg = TemplateNLG(is_user=True)

    # assemble
    # user_agent = PipelineAgent(
    #     user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(
        user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    # set_seed(3407)

    analyzer.comprehensive_analyze(
        sys_agent=sys_agent, model_name='LLM',
        total_dialog=100
    )


if __name__ == '__main__':
    run()