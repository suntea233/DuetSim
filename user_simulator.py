from convlab2.policy.policy import Policy
from convlab2.task.multiwoz.goal_generator import GoalGenerator
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlg.template.multiwoz.nlg import slot2word
from base_model import FlanT5, ChatGPT, ChatGLM2, LLAMA, LLAMA2
from convlab2.dialog_agent import Agent
from convlab2.policy.user_sim.multiwoz.goal import Goal
import copy

word2slot = {value.capitalize(): key.capitalize() for key, value in slot2word.items()}
old_slot_map = {
    'addr': "address",
    'post': "postcode",
    'price': "price range",
    'arrive': "arrive by",
    'leave': "leave at",
    'depart': "departure",
    'dest': "destination",
    'fee': "entrance fee",
    'open': 'open hours',
    # 'car': "type",
    'ticket': 'price',
    'id': 'train id',
    'people': 'book people',
    'stay': 'book stay',
    'time': 'duration',
    'none': '',
}

slot_map = {value.capitalize(): key.capitalize() for key, value in old_slot_map.items()}


class LLM_US(Agent):
    def __init__(self, model_name, print_details=True):
        if model_name == "flant5":
            self.model = FlanT5()

        elif model_name == "chatgpt":
            self.model = ChatGPT()

        elif model_name == "chatglm2":
            self.model = ChatGLM2()

        elif model_name == "llama":
            self.model = LLAMA()

        elif model_name == 'llama2':
            self.model = LLAMA2()
        else:
            raise NotImplementedError

        self.reward_func = lambda goal, dialog, completed: 40 if completed else -20
        self.print_details = print_details
        self.domain_intent = ['inform', 'request']
        self.general_intent = ['thank', 'bye']
        self.policy = SimulatorPolicy(print_details=print_details)
        self.opponent_name = 'sys'
        self.name = 'usr'
        self.usr_nlg = TemplateNLG(is_user=True)
        self.sys_nlg = TemplateNLG(is_user=False)
        self.discriminator = ChatGPT("discriminator")
        self.prompt_for_discriminator = "From now on, you are supposed to play the role of a supervisor. Your task is to assess whether the CUSTOMER's reply meet the CUSTOMER GOAL. 'DA' represents dialog action, which is another form of reply. However, at the same time, when the ASSISTANT initiates a question, the CUSTOMER's reply should address the ASSISTANT's question. Each of the CUSTOMER's reply only needs to fulfill one of the CUSTOMER GOAL. You should be mindful that the CUSTOMER's DA reply should not duplicate what's already in the CONTEXT. "
        # self.prompt_for_discriminator = 'You are now in the role of a discriminator; your task is to assess whether the reply  CUSTOMER GOAL means that '

        # self.supervisor = copy.deepcopy(self.model)
        #
        # self.generator = copy.deepcopy(self.model)

    def convert_old_2_nlg(self, observation):
        temp = copy.deepcopy(observation)
        for i in range(len(temp)):
            if temp[i][2] in word2slot:
                temp[i][2] = word2slot[temp[i][2]]
        return temp

    def convert_old_2_slot(self, observation):
        temp = copy.deepcopy(observation)
        for i in range(len(temp)):
            if temp[i][1].lower() == 'taxi' and temp[i][2].lower() == 'type':
                temp[i][2] = 'Car'
            if temp[i][2] in slot_map:
                temp[i][2] = slot_map[temp[i][2]]
        return temp

    def convert_2_old_slot(self, observation):
        temp = copy.deepcopy(observation)
        for i in range(len(temp)):
            if temp[i][2].lower() in old_slot_map:
                temp[i][2] = old_slot_map[temp[i][2].lower()].capitalize()
        return temp

    def check_and_return(self, text, mappings):
        for key in mappings.keys():
            if key in text:
                return mappings[key]
        return 'none'

    def create_mappings(self, n):
        mappings = {}
        for i in range(n):
            k1 = chr(ord('A') + i) + ":"
            value = chr(ord('A') + i)
            k2 = chr(ord('A') + i) + "."
            mappings[k1] = value
            mappings[k2] = value
        return mappings

    def parse_ans(self, text, choice, raw_list=None):
        if raw_list:
            if text:
                raw_text = text[0].upper()

                raw_text = raw_text.replace(".", "")
                try:
                    return [raw_list[ord(raw_text) - 65]]
                except:
                    mappings = self.create_mappings(len(raw_list))
                    ans = self.check_and_return(text, mappings)
                    if ans != 'none':
                        return [raw_list[ord(ans) - 65]]
                    else:
                        return ['none']
            else:
                return ['thank']
        else:
            import re
            text = text[0].upper()
            text = text.replace(".", "")
            pattern = r'{}\.(\w+)'.format(text)
            matches = re.findall(pattern, choice)
            return matches

    def get_intent(self, multi_choice, intents, prompt, supervisor_ans=""):
        alphabet = ' or '.join([f"{chr(65 + i)}" for i in range(len(intents))])
        # intent_prompt = 'You need to respond by answering multiple choice questions about the INTENT, '
        # intent_prompt = 'Which INTENT will you choose to fulfill the GOAL and context?\n'
        intent_prompt = """\n\n[REQUIREMENTS]\n"INTENT" refers to the CUSTOMER's purpose or action expressed during the CONTEXT.\nAt the beginning of a CONTEXT, "inform" must be used as a the INTENT.\n If the ASSISTANT is requesting, YOU MUST ANSWER "INFORM" INSTEAD OF "REQUEST!\n\n[QUESTION]\nWhich INTENT will you choose according to the GOAL?\n"""
        # intent_prompt = """\n\n"INTENT" refers to the CUSTOMER's purpose or action expressed during the CONVERSATION.\n\nWhich INTENT will you choose according to the GOAL and CONVERSATION? \n"""
        # intent_prompt = """\nSelect the appropriate INTENT based on the CUSTOMER's GOAL, where "INTENT" refers to the purpose or action the customer expresses at the start of the conversation. At the beginning of a conversation, "inform" must be used as a the INTENT.\n If the ASSISTANT needs information, it should use "inform" instead of "request"."""
        message = supervisor_ans + prompt + intent_prompt + multi_choice + "\n\nJUST ANSWER {}.\n".format(alphabet)
        intent = self.model.generate(message)
        ans = self.parse_ans(text=intent, choice=multi_choice,raw_list=intents)
        if ans == ['none']:
            ans = ['thank']
        return ans


    def get_domain(self, multi_choice, intent, domains, prompt,supervisor_ans=''):
        alphabet = ' or '.join([f"{chr(65 + i)}" for i in range(len(domains))])
        # domain_prompt = 'You need to respond by answering multiple choice questions about the DOMAIN, '
        # domain_prompt = 'Which DOMAIN will you choose to fulfill the GOAL and context?\n'
        domain_prompt = """\n\n[REQUIREMENTS]\n"DOMAIN" refers to a specific subject area or category that encompasses a set of related tasks, intent. CUSTOMER's intent is "{}".\n\nWhich DOMAIN will you choose according to the GOAL and CONVERSATION?\n""".format(
            intent)
        # domain_prompt = """\nSelect the appropriate DOMAIN based on the CUSTOMER's GOAL and CONTEXT, where "DOMAIN" pertains to a distinct subject area or category that includes a range of interconnected tasks and intents. CUSTOMER's intent is "{}".\n""".format(intent)
        message = supervisor_ans + prompt + domain_prompt+ multi_choice +  "\nJUST ANSWER {}.\n".format(alphabet)
        domain = self.model.generate(message)
        ans = self.parse_ans(text=domain, choice=multi_choice,raw_list=domains)
        if ans == ['none']:
            ans = ['general']
        return ans


    def get_slot(self, multi_choice, intent, domain, slots, prompt,supervisor_ans=''):
        alphabet = ' or '.join([f"{chr(65 + i)}" for i in range(len(slots))])
        # slot_prompt = 'You need to respond by answering multiple choice questions about the SLOT, '
        # slot_prompt = 'Which SLOT will you choose to fulfill the GOAL and context?\n'
        slot_prompt = """\n\n[REQUIREMENTS]\n"SLOT" refers to a specific piece of information or data that is relevant to a CUSTOMER's intent within a CONVERSATION. CUSTOMER's intent is "{}", domain is "{}".\n\nWhich SLOT will you choose according to the GOAL and CONVERSATION?\nYou have to mentioned other slot before "book".""".format(
            intent, domain)
        # slot_prompt = """\nDetermine the appropriate SLOT based on the CUSTOMER's GOAL and CONTEXT, where "SLOT" pertains to a specific piece of information or data that is relevant to the customer's intent within the conversation. CUSTOMER's intent is "{}", domain is "{}".\n""".format(intent,domain)

        message = supervisor_ans + prompt + slot_prompt + multi_choice + "\nJUST ANSWER {}.\n".format(alphabet)
        slot = self.model.generate(message)
        ans = self.parse_ans(text=slot, choice=multi_choice, raw_list=slots)
        return ans


    def get_value(self, multi_choice, intent, domain, slot, values, prompt,supervisor_ans=''):
        alphabet = ' or '.join([f"{chr(65 + i)}" for i in range(len(values))])
        # value_prompt = 'You need to respond by answering multiple choice questions about the VALUE, '
        # value_prompt = "Which VALUE will you choose to fulfill the GOAL and context?\n"
        value_prompt = """\n\n[REQUIREMENTS]\n"VALUE" refers to the specific content or data that fills a particular slot. CUSTOMER's intent is "{}", domain is "{}", slot is "{}".\n\nWhich VALUE will you choose according to the GOAL and CONVERSATION?\n""".format(
            intent, domain, slot)
        # value_prompt = """\nSelect the appropriate VALUE based on the CUSTOMER's GOAL and CONTEXT, where "VALUE" represents the specific content or data that occupies a specific slot. CUSTOMER's intent is "{}", domain is "{}", slot is "{}".\n""".format(intent,domain,slot)

        message = supervisor_ans + prompt + value_prompt + multi_choice + "\nJUST ANSWER {}.\n".format(alphabet)
        value = self.model.generate(message)
        if value in multi_choice:
            try:
                ans = self.parse_ans(text=value, choice=multi_choice, raw_list=values)
            except:
                ans = [value]
        else:
            ans = ['none']

        if ans == []:
            ans = ['none']
        return ans


    def init_session(self, **kwargs):
        self.usr_nlg.init_session()
        self.sys_nlg.init_session()
        self.policy.init_session(**kwargs)
        self.history = []
        self.da_history = []
        # self.prompt = "GOAL:{}\nNow you are pretending as a CUSTOMER, according the context to fulfill the GOAL.\n"
        self.goal_status = self.policy.goal.get_goal_list()
        self.goal_status = [[item.replace('info', 'inform').replace('reqt', 'request') for item in inner_list] for
                            inner_list in self.goal_status]
        self.demonstration = ''


    def _get_intent(self, prompt, supervisor_ans=''):
        intents = list(set([item[0] for item in self.complete if item[-1] != "fulfilled"])) + self.general_intent
        intent_choice = ' '.join([f"{chr(65 + i)}.{item}" for i, item in enumerate(intents)])
        temp_intent = self.get_intent(intent_choice, intents, prompt, supervisor_ans)
        try:
            intent = temp_intent[0]
        except:
            intent = 'thank'
        return intent


    def _get_domain(self, intent, prompt,supervisor_ans=''):
        # get domain
        domains = list(set([item[1] for item in self.complete if item[0] == intent and item[-1] != "fulfilled"]))
        if len(domains) <= 1 and domains != []:
            domain = domains[0]
        elif domains == []:
            return []
        else:
            self.domain_choice = ' '.join([f"{chr(65 + i)}.{item}" for i, item in enumerate(domains)])
            domain = self.get_domain(self.domain_choice, intent, domains, prompt,supervisor_ans)
            if domain == []:
                domain = 'none'
            else:
                domain = domain[0]
        return domain

    def _get_slot(self, intent, domain, prompt,supervisor_ans=''):
        slots = list(set([item[2] for item in self.complete if
                          item[0] == intent and item[1] == domain and item[-1] != "fulfilled"]))

        try:
            if len(slots) == 1:
                return slots[0]
        except:
            print(1)

        if slots == []:
            return 'none'

        slot_choice = " ".join([f"{chr(65 + i)}.{item}" for i, item in enumerate(slots)])

        slot = self.get_slot(slot_choice, intent, domain, slots, prompt,supervisor_ans)

        if slot == []:
            slot = 'none'
        else:
            slot = slot[0]
        return slot

    def _get_value(self, intent, domain, slot, prompt,supervisor_ans=''):
        if slot == "none":
            return 'none'

        elif intent.lower() == "request":
            return "?"
        else:
            values = list(set([item[3] for item in self.complete if
                               item[0] == intent and item[1] == domain and slot in item[
                                   2] and item[-1] != "fulfilled"]))

            if len(values) <= 1:
                value = values[0]
            else:
                value_choice = " ".join([f"{chr(65 + i)}.{item}" for i, item in enumerate(values)])
                value = self.get_value(value_choice, intent, domain, slot, values, prompt,supervisor_ans)[0]
            return value


    def get_full_action(self, prompt, supervisor_ans=''):
        # get intent

        intent = self._get_intent(prompt, supervisor_ans)

        if intent in self.domain_intent:
            # get_domain
            domain = self._get_domain(intent, prompt,supervisor_ans)
            # get slot
            slot = self._get_slot(intent, domain, prompt,supervisor_ans)
            # get value
            value = self._get_value(intent, domain, slot, prompt,supervisor_ans)

            return [[intent.capitalize(), domain.capitalize(), slot.capitalize(), value.capitalize()]]

        else:
            return [[intent.capitalize(), 'general', 'none', 'none']]


    def get_supervisor_answer(self, cnt, observation, da_response):
        if cnt > 4:
            return True
        self.supervisor_prompt = """[INSTRUCTION]Now you will play the role of a supervisor to determine which option is correct. CUSTOMER's and ASSISTANT's replies are made by [[INTENT,DOMAIN,SLOT,VALUE]].

[CONVERSATION]
ASSISTANT:{}
CUSTOMER:{}

[CHOICES]
A. when the domain changed, the CUSTOMER doesn't start with 'INFORM'.
B. When the assistant asks a question, the customer does not respond to the assistant's question.
C. The customer's response is correct.

The correct answer is
""".format(observation, da_response)

        answer = self.model.generate(self.supervisor_prompt)


        solutions = ["\nYou must start with 'inform'\n", "\nYou must answer the ASSISTANT's question\n", True]
        raw_text = answer[0].upper()
        raw_text = raw_text.replace(".", "")
        try:
            return solutions[ord(raw_text) - 65]
        except:
            mappings = self.create_mappings(len(solutions))
            ans = self.check_and_return(raw_text, mappings)
            if ans != 'none':
                return solutions[ord(ans) - 65]
            else:
                return solutions[-1]



    def response(self, observation):
        temp_observation = self.convert_2_old_slot(observation)

        self.goal_status = self.policy.goal.get_goal_list()
        self.goal_status = [[item.replace('info', 'inform').replace('reqt', 'request') for item in inner_list] for
                            inner_list in self.goal_status]

        self.input_action = observation

        if observation:
            sys_observation = self.convert_old_2_nlg(observation)
            sys_observation = self.sys_nlg.generate(sys_observation)
        else:
            sys_observation = "Hello, can I help you?"

        self.da_history.append([self.opponent_name, observation])
        self.history.append([self.opponent_name, sys_observation])

        self.complete = [status for status in self.goal_status if status[-1] != "fulfilled"]
        judge = [status for status in self.goal_status if status[-1] != "fulfilled"]

        self.complete = [a_entry for a_entry in self.complete if not (a_entry[0] == 'request' and any(
            a_entry[1:3] == [b_entry[1].lower(), b_entry[2].lower()] for b_entry in temp_observation))]

        self.user_status = [status[:4] for status in self.goal_status]

        if judge == []:
            temp_da_response = [['thank', 'general', 'none', 'none']]
            nlg_response = self.convert_old_2_nlg(temp_da_response)
            usr_response = self.usr_nlg.generate(nlg_response)
        else:
            # self.prompt = """Now you are pretending as a CUSTOMER, according the CONVERSATION to fulfill the GOAL.\n[GOAL]\n{}\n\nCUSTOMER's reply is made by [[INTENT,DOMAIN,SLOT,VALUE]]\n"""
            self.prompt = """Now you are pretending as a CUSTOMER, according the CONVERSATION to fulfill the GOAL.\n\nCUSTOMER's reply is made by [[INTENT,DOMAIN,SLOT,VALUE]]\n"""
            self.prompt = self.prompt.format(self.complete)

            self.context = '\n'.join(': '.join(inner_list) for inner_list in self.history).replace('usr',
                                                                                                   'CUSTOMER').replace(
                'sys', 'ASSISTANT')

            formatted = self.format_context()

            self.prompt = self.prompt
            # self.prompt = self.prompt + "\n[CONVERSATION]\n" + self.context
            # self.prompt = self.prompt + "\n[CONVERSATION]\nASSISTANT:" + sys_observation

            temp_da_response = self.get_full_action(self.prompt)

            cnt = 0
            nlg_response = self.convert_old_2_nlg(temp_da_response)
            usr_response = self.usr_nlg.generate(nlg_response)


            supervisor_ans = self.get_supervisor_answer(cnt, observation, temp_da_response)

            temp_prompt = copy.deepcopy(self.prompt)

            while supervisor_ans != True:
                cnt += 1
                supervised_da_response = self.get_full_action(temp_prompt, supervisor_ans)
                supervisor_ans = self.get_supervisor_answer(cnt, observation, supervised_da_response)

                if supervisor_ans == True:
                    temp_da_response = supervised_da_response


        da_response = self.convert_old_2_slot(temp_da_response)

        self.da_history.append([self.name, da_response])
        self.output_action = da_response



        if self.output_action[0][0].lower() != 'thank' and self.output_action[0][0].lower() != 'bye':
            self.natural_language = self.get_nlg(sys_observation,temp_da_response)
        elif self.output_action[0][0].lower() == 'thank':
            natural_language = "Thank you for your detailed answering!"
            self.natural_language = self.get_second_nlg(sys_observation,natural_language)
        else:
            natural_language = "Goodbye, thank you for your kind reply!"
            self.natural_language = self.get_second_nlg(sys_observation,natural_language)

        self.sys_observation = sys_observation
        
        self.history.append([self.name, self.natural_language])
        
        self.policy.predict(
            sys_act=self.input_action, usr_act=self.output_action)

        return self.output_action

    def get_nlg(self, context, output_action):
        init_prompt = """[EXAMPLE]
[['inform', 'restaurant', 'book day', 'Tuesday']]
The restaurant is booked on Tuesday.
[END EXAMPLE]

{}
Please translate the list into natural language.
""".format(output_action)

        natural_language = self.model.generate(init_prompt)
        return self.get_second_nlg(context,natural_language)


    def get_second_nlg(self, context,natural_language):
        second_prompt = """Conversation: \n{}\n\nSENTENCE: {}\nBased on conversation, play the role of CUSTOMER and rewrite this sentence to make it smoother, more advanced, and more conversational, about 20 words.""".format(
            self.context,natural_language)

        ans = self.model.generate(second_prompt)
        return ans


    def format_da(self, da):
        result = []

        for item in da:
            if item[0] == 'usr':
                result.append('CUSTOMER:' + str(item[1]))
            elif item[0] == 'sys':
                result.append('ASSISTANT:' + str(item[1]))
            else:
                result.append(item[0] + ':' + str(item[1]))

        final_result = '\n'.join(result)
        return final_result


    def format_context(self):
        context = []
        for da_history, nl_history in zip(self.da_history, self.history):
            role = da_history[0]
            if role == 'sys':
                role = 'ASSISTANT:'
            else:
                role = 'CUSTOMER:'
            da = da_history[1]
            nl = nl_history[1]
            dialogue = " ".join([role, nl, "DA:", str(da)])
            context.append(dialogue)
        return "\n".join(context)

    def get_in_da(self):
        return self.input_action

    def get_out_da(self):
        return self.output_action

    def is_terminated(self):
        if hasattr(self.policy, 'is_terminated'):
            return self.policy.is_terminated()
        return None


class SimulatorPolicy(Policy):
    def __init__(self, print_details=True):
        super().__init__()
        self.print_details = print_details
        self.goal_generator = GoalGenerator()
        self.max_turn = 40
        self.max_initiative = 4
        self.sys_acts = []


    def get_goal(self):
        return self.domain_goals


    def reset_turn(self):
        self.__turn = 0


    def init_session(self, goal=None):
        self.reset_turn()
        if not goal:
            self.goal = Goal(
                goal_generator=self.goal_generator,
                print_details=self.print_details
            )
            self.goal_in_natural_language = self.goal_generator.build_message(
                self.goal.goal_to_transform)
        else:
            self.goal = goal
            self.goal_in_natural_language = ''

        self.domain_goals = self.goal.domain_goals
        self.sys_acts = []
        self.usr_acts = []
        self.terminated = False
        self.mode = "semantic"
        self.time_step = 0
        self.max_history = 4


    def _no_offer(self, system_in):
        for intent, domain, slot, value in system_in:
            if intent.lower() == "nooffer":
                self.terminated = True
                return True
            else:
                return False

    # from tus
    def predict(self, sys_act, usr_act):
        # allow_general_intent = False
        # self.model.eval()

        # if not self.add_sys_from_reward:
        self.goal.update_user_goal(action=sys_act, char="sys")
        self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        self.semantic_action = usr_act
        # TODO
        if self.is_finish():
            self.semantic_action, self.utterance = self._good_bye()

        self.goal.update_user_goal(action=self.semantic_action, char="usr")
        # self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)

        if self.mode == "language":
            # print("in", sys_act)
            # print("out", self.utterance)
            return self.utterance
        else:
            return self.semantic_action


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
        else:
            return [["bye", "general", "None", "None"]], "bye"


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


    def _usr_terminate(self):
        for act in self.semantic_action:
            if act[0] in ['thank', 'bye']:
                return True
        return False
