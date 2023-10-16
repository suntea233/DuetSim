from convlab2.dialog_agent import Agent
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlg.generative_models.flanT5 import FLANT5Model
import random
from copy import deepcopy

# imports for policy
import json
from convlab2.policy.user_sim.multiwoz.goal import slot_name_map
from convlab2.policy.policy import Policy
from convlab2.policy.user_sim.multiwoz.goal import Goal
from convlab2.task.multiwoz.goal_generator import GoalGenerator


class UserSimulatorE2E(Agent):
    def __init__(self,
                 name='user', shots_file='data/multiwoz/train_modified.json',
                 sample_based_on_user_goal=True, num_shots=2,
                 nlg=None, print_details=True,
                 task_description_file_path=None,
                 use_bullet_pointed_goal=True,
                 temperature=0.8,
                 sys_nlu=True,
                 debug_prompt_processing=False):
        super(UserSimulatorE2E, self).__init__(name=name)
        self.nlu = BERTNLU(  # to predict user's utterances
            mode='all',
            config_file='multiwoz_all_context.json',
            model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_all_context.zip')  # noqa

        self.sys_nlu = sys_nlu
        if self.sys_nlu:
            self.usr_nlu = BERTNLU(  # to predict utterances coming from sys
                mode='sys',
                config_file='multiwoz_sys_context.json',
                model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_sys_context.zip')

        self.opponent_name = 'sys'

        self.nlg = nlg if nlg else FLANT5Model()

        self.policy = SimulatorPolicy(
            shots_file=shots_file,
            print_details=print_details)
        self.history = []
        self.num_shots = num_shots
        self.sample_based_on_user_goal = sample_based_on_user_goal
        self.shots = []
        self.prompt = ''
        self.input_action = None
        self.output_action = None
        self.print_details = print_details
        self.task_description_file_path = task_description_file_path
        self.debug_prompt_processing = debug_prompt_processing
        self.use_bullet_pointed_goal = use_bullet_pointed_goal
        self.temperature = temperature

    def _retrieve_task_description(self, user_goal_domains):
        task_description = open(self.task_description_file_path, 'r').read()

        if '{}' in task_description:
            if len(user_goal_domains) == 0:
                # don't mention any domain
                user_goal_domains_text = ''
            elif len(user_goal_domains) == 1:
                # for restaurant domain
                user_goal_domains_text = ' about: ' + user_goal_domains[0]
            else:
                # for restaurant, attraction and hotel domains
                user_goal_domains_text = (
                    ' about: ' + ', '.join(user_goal_domains[:-1]) +
                    ' and ' + user_goal_domains[-1]
                )
            task_description = task_description.format(user_goal_domains_text)

        return task_description + '\n'

    @staticmethod
    def generate_prompt_from_shot(
        shot,
        shot_number,
        use_bullet_pointed_goal=True
    ):

        def _preprocessing_text(text):
            return (
                text.replace("<span class='emphasis'>", '')
                    .replace('</span>', '') + ".")\
                    .replace('..', '.')

        if isinstance(shot, dict):
            message = shot.get('goal', {}).get('message', [])
            log = shot.get('log')
        else:
            message = shot
            log = None

        prompt = '\nExample {}:\nREQUIREMENTS:'.format(shot_number + 1)

        # user goal
        if type(message) == list:
            message = [_preprocessing_text(m) for m in message]
            message = ' '.join(message).split('.')[:-1]
            if use_bullet_pointed_goal:
                prompt += '\n'
                for i, sentence in enumerate(message):
                    if i == 0:
                        sentence = ' ' + sentence
                    sentence = '-' + sentence + '.'
                    if i != len(message) - 1:
                        sentence += '\n'
                    prompt += sentence
            else:
                prompt += ' ' + '.'.join(message)
        else:
            prompt += _preprocessing_text(message)

        prompt += '\nCONVERSATION:'

        # conversation
        conversation = []
        if log is not None:
            prompt += '\n'
            for agent_idx, agent_turn in enumerate(log):
                if agent_idx % 2 == 0:
                    agent_name = 'CUSTOMER: '
                else:
                    agent_name = 'ASSISTANT: '
                conversation.append(agent_name + agent_turn.get('text'))
            prompt += '\n'.join(conversation)

        return prompt

    def init_session(self, **kwargs):
        self.nlu.init_session()
        if self.sys_nlu:
            self.usr_nlu.init_session()
        self.policy.init_session(**kwargs)
        self.nlg.init_session()
        self.history = []
        self.shots = []

        if hasattr(self.policy, 'goal_in_natural_language') and hasattr(
                self.policy, 'sample_convs'):

            self.sample_shots_based_on_user_goal_attributes()
            if len(self.shots) < self.num_shots:
                self.shots = random.sample(
                    list(self.policy.sample_convs.values()), self.num_shots)

            # get the domains from the shots
            user_goal_domains = set()
            for shot in self.shots:
                for domain_name, domain_val in shot.get('goal').items():
                    if domain_val and domain_name not in ['message', 'topic']:
                        user_goal_domains.add(domain_name)
            user_goal_domains = list(user_goal_domains)

            self.prompt = self._retrieve_task_description(user_goal_domains)

            prompt_tmp = []
            for i, shot in enumerate(self.shots):

                prompt = self.generate_prompt_from_shot(
                    shot=shot, shot_number=i,
                    use_bullet_pointed_goal=self.use_bullet_pointed_goal)
                prompt_tmp.append(prompt)

            self.prompt += '\n'.join(prompt_tmp)

            prompt = self.generate_prompt_from_shot(
                shot=self.policy.goal_in_natural_language[1],
                shot_number=len(self.shots),
                use_bullet_pointed_goal=self.use_bullet_pointed_goal)
            self.prompt += ('\n' + prompt)

            print(self.prompt)

    def response(self, observation):
        if self.sys_nlu:
            # get dialog acts from observation (sys dialog acts)
            self.input_action = self.usr_nlu.predict(  # or sys_nlu
                observation, context=[x[1] for x in self.history[:-1]])
        else:
            self.input_action = self.nlu.predict(  # or sys_nlu
                observation, context=[x[1] for x in self.history[:-1]])
        self.input_action = deepcopy(self.input_action)  # get rid of reference problem  # noqa

        # update user goal given sys dialog acts

        # update prompt (i.e. append sys text) and call nlg
        if observation:
            self.prompt = self.prompt + "\nASSISTANT: " + str(observation)
        model_response = self.nlg.generate(
            self.prompt, temperature=self.temperature,
            debug_prompt_processing=self.debug_prompt_processing)

        self.prompt += "\nCUSTOMER: " + model_response

        self.history.append([self.opponent_name, observation])
        self.history.append([self.name, model_response])

        # get dialog acts from what user has said
        self.output_action = self.nlu.predict(
            model_response, context=[x[1] for x in self.history[:-1]])
        if self.print_details:
            print("CUSTOMER:", model_response, 'DA:', self.output_action)
        else:
            print("CUSTOMER:", model_response)

        self.policy.predict(
            sys_act=self.input_action, usr_act=self.output_action)

        # update user goal given user dialog acts

        # return nlg
        return model_response

    def get_in_da(self):
        return self.input_action

    def get_out_da(self):
        return self.output_action

    def is_terminated(self):
        if hasattr(self.policy, 'is_terminated'):
            return self.policy.is_terminated()
        return None

    def sample_shots_based_on_user_goal_attributes(self):
        if self.sample_based_on_user_goal:
            sample_convs = list(self.policy.sample_convs.values())
            random.shuffle(sample_convs)
            id2similarity = {i: 0 for i in range(len(sample_convs))}

            target_goal_domains = set(self.policy.goal.raw_goal.keys())
            target_goal_slots = set()
            for dom, dom_vals in self.policy.goal.raw_goal.items():
                for intent, vals in dom_vals.items():
                    if type(vals) == dict:
                        for val in vals.keys():
                            target_goal_slots.add('__'.join([dom, intent, val]))
            for conv_id, conv in enumerate(sample_convs):
                conv_goal_domains = set()
                conv_goal_slots = set()
                if type(conv.get('goal').get('message')) == list:
                    # if the user goal can be split in sentences, continue
                    for domain_name, domain_val_to_compare in conv.get('goal').items():  # noqa
                        if domain_val_to_compare and domain_name not in {
                                'message', 'topic'}:
                            conv_goal_domains.add(domain_name)
                            for intent, slots_to_compare in domain_val_to_compare.items():  # noqa
                                if type(slots_to_compare) == list:
                                    mapped_slots_to_compare = {
                                        slot_name_map.get(
                                            k, k)  # k itself as default
                                        for k in slots_to_compare}
                                elif type(slots_to_compare) == dict:
                                    mapped_slots_to_compare = {
                                        slot_name_map.get(k, k)
                                        for k in slots_to_compare.keys()}
                                else:
                                    mapped_slots_to_compare = {}
                                for slot_name in mapped_slots_to_compare:
                                    conv_goal_slots.add('__'.join(
                                        [domain_name, intent, slot_name]))
                jacc_domains = len(target_goal_domains.intersection(
                    conv_goal_domains)) / len(target_goal_domains.union(
                        conv_goal_domains))
                jacc_slots = len(target_goal_slots.intersection(
                    conv_goal_slots)) / len(target_goal_slots.union(
                        conv_goal_slots))
                id2similarity[conv_id] = jacc_domains * jacc_slots

                # print('a', target_goal_slots, target_goal_domains)
                # print('b', conv_goal_slots, conv_goal_domains)
                # print(id2similarity[conv_id])

            id2similarity = {
                k: v for k, v in sorted(
                    id2similarity.items(), key=lambda item: -item[1])}
            count = 0
            for k, v in id2similarity.items():
                count += 1
                if count > self.num_shots:
                    break
            shot_conv_ids = list(id2similarity)[:self.num_shots]
            self.shots = [sample_convs[conv_id] for conv_id in shot_conv_ids]

            print("selected shots based on user goal similarity scores")
            print("--------------------\n")


class SimulatorPolicy(Policy):
    def __init__(self, shots_file, print_details=True):
        super(SimulatorPolicy, self).__init__()
        self.max_turn = 40
        self.max_initiative = 4
        self.goal_generator = GoalGenerator()
        self.sample_convs = json.load(open(shots_file, 'r'))
        self.__turn = 0
        self.goal = None
        self.agenda = None
        self.sys_acts = []
        self.print_details = print_details

    def get_goal(self):
        return self.domain_goals

    def init_session(self, ini_goal=None):
        """ Build new Goal and Agenda for next session """
        self.reset_turn()
        if not ini_goal:
            self.goal = Goal(
                goal_generator=self.goal_generator,
                print_details=self.print_details)
            self.goal_in_natural_language = self.goal_generator.build_message(
                self.goal.goal_to_transform)
        else:
            self.goal = ini_goal
            self.goal_in_natural_language = ''
        self.domain_goals = self.goal.domain_goals

        # from tus
        '''
        self.mentioned_domain = []
        self.time_step = 0
        self.topic = 'NONE'
        remove_domain = "police"  # remove police domain in inference

        if not goal:
            self.new_goal(remove_domain=remove_domain)
        else:
            self.read_goal(goal)

        # print(self.goal)
        if self.config.get("reorder", False):
            self.predict_action_list = self.goal.action_list()
        else:
            self.predict_action_list = self.action_list
        self.sys_history_state = None  # to save sys history

        self.feat_handler.initFeatureHandeler(self.goal)
        self.pre_usr_act = None
        '''
        self.sys_acts = []
        self.usr_acts = []
        self.terminated = False
        self.mode = "semantic"
        self.time_step = 0
        self.max_history = 4

    # from tus
    '''
    def transform_usr_act(self, usr_output, action_list, mode="max"):
        is_finish, usr_action = self._finish_conversation()
        if is_finish:
            self.terminated = True
            return usr_action

        usr_action = self._get_acts(
            usr_output, action_list, mode)

        # if usr_action is empty, sample at least one
        while not usr_action:
            usr_action = self._get_acts(
                usr_output, action_list, mode="pick-one")

        if self.use_domain_mask:
            domain_mask = self._get_prediction_domain(torch.round(
                torch.sigmoid(usr_output[0, 0, :])).tolist())
            usr_action = self._mask_user_action(usr_action, domain_mask)

        return usr_action
    '''

    # from tus
    def _no_offer(self, system_in):
        for intent, domain, slot, value in system_in:
            if intent.lower() == "nooffer":
                self.terminated = True
                return True
            else:
                return False

    # def is_terminated(self):
    #     # Is there any action to say?
    #     return self.terminated

    def reset_turn(self):
        self.__turn = 0

    # from tus
    def predict(self, sys_act, usr_act):
        # allow_general_intent = False
        # self.model.eval()

        # if not self.add_sys_from_reward:
        self.goal.update_user_goal(action=sys_act, char="sys")
        self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        # history = []
        # if self.usr_acts:
        #     if self.max_history == 1:
        #         history = self.usr_acts[-1]
        #     else:
        #         history = self.usr_acts[-1 * self.max_history:]  # noqa
        # inputs = json.dumps({"system": sys_act,
        #                      "goal": self.goal.get_goal_list(),
        #                      "history": history,
        #                      "turn": str(int(self.time_step / 2))})
        # with torch.no_grad():
        #     raw_output = self._generate_action(
        #         raw_inputs=inputs, mode=mode, allow_general_intent=allow_general_intent)  # noqa
        # output = self._parse_output(raw_output)
        # self.semantic_action = self._remove_illegal_action(output["action"])
        # if not self.only_action:
        #     self.utterance = output["text"]

        self.semantic_action = usr_act
        # TODO
        if self.is_finish():
            self.semantic_action, self.utterance = self._good_bye()

        # if self.is_finish():
        #     print("terminated")

        # if self.is_finish():
        #     good_bye = self._good_bye()
        #     self.goal.add_usr_da(good_bye)
        #     return good_bye

        self.goal.update_user_goal(action=self.semantic_action, char="usr")
        # self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)

        # if self._usr_terminate(usr_action):
        #     print("terminated by user")
        #     self.terminated = True

        # del inputs

        if self.mode == "language":
            # print("in", sys_act)
            # print("out", self.utterance)
            return self.utterance
        else:
            return self.semantic_action

    def _usr_terminate(self):
        for act in self.semantic_action:
            if act[0] in ['thank', 'bye']:
                return True
        return False

    def is_finish(self):
        # stop by model generation?
        if self._finish_conversation_rule():
            self.terminated = True
            return True
        elif self._usr_terminate():
            self.terminated = True
            return True
        self.terminated = False
        return False

    def is_success(self):
        task_complete = self.goal.task_complete()
        # goal_status = self.goal.all_mentioned()
        # should mentioned all slots
        if task_complete:  # and goal_status["complete"] > 0.6:
            return True
        return False

    def _good_bye(self):
        if self.is_success():
            return [['thank', 'general', 'none', 'none']], "thank you. bye"
            # if self.mode == "semantic":
            #     return [['thank', 'general', 'none', 'none']]
            # else:
            #     return "bye"
        else:
            return [["bye", "general", "None", "None"]], "bye"
            # if self.mode == "semantic":
            #    return [["bye", "general", "None", "None"]]
            # return "bye"

    def _finish_conversation_rule(self):
        if self.is_success():
            return True

        if self.time_step > self.max_turn:
            return True

        if (len(self.sys_acts) > 4) and (
            self.sys_acts[-1] == self.sys_acts[-2]) and (
                self.sys_acts[-2] == self.sys_acts[-3]):  # noqa
            return True
        return False

    def is_terminated(self):
        # Is there any action to say?
        self.is_finish()
        return self.terminated
