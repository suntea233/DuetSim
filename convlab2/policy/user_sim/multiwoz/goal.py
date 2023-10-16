"""
The user goal for unify data format - copied from GenTUS
"""
import json
from random import shuffle
# from convlab2.policy.tus.unify.Goal import old_goal2list
# from convlab2.task.multiwoz.goal_generator import GoalGenerator
from convlab2.policy.rule.multiwoz.policy_agenda_multiwoz import Goal as ABUS_Goal
from convlab2.util.custom_util import slot_mapping
DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, ""]

NOT_MENTIONED = "not mentioned"
FULFILLED = "fulfilled"
REQUESTED = "requested"
CONFLICT = "conflict"




# copy from https://github.com/ConvLab/ConvLab-3/blob/master/convlab/policy/tus/unify/util.py
slot_name_map = {
    'addr': "address",
    'post': "postcode",
    'pricerange': "price range",
    'arrive': "arrive by",
    'arriveby': "arrive by",
    'leave': "leave at",
    'leaveat': "leave at",
    'depart': "departure",
    'dest': "destination",
    'fee': "entrance fee",
    'open': 'open hours',
    'car': "type",
    'car type': "type",
    'ticket': 'price',
    'trainid': 'train id',
    'id': 'train id',
    'people': 'book people',
    'stay': 'book stay',
    'none': '',
    'attraction': {
        'price': 'entrance fee'
    },
    'hospital': {},
    'hotel': {
        'day': 'book day', 'price': "price range"
    },
    'restaurant': {
        'day': 'book day', 'time': 'book time', 'price': "price range"
    },
    'taxi': {},
    'train': {
        'day': 'day', 'time': "duration"
    },
    'police': {},
    'booking': {}
}


def old_goal2list(goal: dict, reorder=False) -> list:
    goal_list = []
    for domain in goal:
        for slot_type in ['info', 'book', 'reqt']:
            if slot_type not in goal[domain]:
                continue
            temp = []
            for slot in goal[domain][slot_type]:
                s = slot
                if slot in slot_name_map:
                    s = slot_name_map[slot]
                elif slot in slot_name_map[domain]:
                    s = slot_name_map[domain][slot]
                # domain, intent, slot, value
                if slot_type in ['info', 'book']:
                    i = 'info' # "inform"
                    v = goal[domain][slot_type][slot]
                else:
                    i = 'reqt' # "request"
                    v = DEF_VAL_UNK
                s = slot_mapping.get(s, s)
                temp.append([domain, i, s, v])
            shuffle(temp)
            goal_list = goal_list + temp
    # shuffle_goal = goal_list[:1] + sample(goal_list[1:], len(goal_list)-1)
    # return shuffle_goal
    return goal_list


class Goal:
    """ User Goal Model Class. """

    def __init__(self, goal=None, goal_generator=None, print_details=True):
        """
        create new Goal from a dialog or from goal_generator
        Args:
            goal: can be a list (create from a dialog), an abus goal, or none
        """
        self.domains = []
        self.domain_goals = {}
        self.status = {}
        self.invert_slot_mapping = {v: k for k, v in slot_mapping.items()}
        self.raw_goal = None

        self._init_goal_from_data(goal, goal_generator)
        self._init_status()
        self.print_details = print_details

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def _init_goal_from_data(self, goal=None, goal_generator=None):
        if not goal and goal_generator:
            goal = ABUS_Goal(goal_generator)
            self.goal_to_transform = goal
            self.raw_goal = goal.domain_goals
            goal = old_goal2list(goal.domain_goals)

        elif isinstance(goal, dict):
            self.raw_goal = goal
            goal = old_goal2list(goal)

        elif isinstance(goal, ABUS_Goal):
            self.raw_goal = goal.domain_goals
            goal = old_goal2list(goal.domain_goals)

        # else:
        #     print("unknow goal")

        # be careful of this order
        for domain, intent, slot, value in goal:
            if domain == "none":
                continue
            if domain not in self.domains:
                self.domains.append(domain)
                self.domain_goals[domain] = {}
            if intent not in self.domain_goals[domain]:
                self.domain_goals[domain][intent] = {}

            if not value:
                if intent == "reqt":
                    self.domain_goals[domain][intent][slot] = DEF_VAL_UNK
                else:
                    print(
                        f"unknown no value intent {domain}, {intent}, {slot}")
            else:
                self.domain_goals[domain][intent][slot] = value

    def _init_status(self):
        for domain, domain_goal in self.domain_goals.items():
            if domain not in self.status:
                self.status[domain] = {}
            for slot_type, sub_domain_goal in domain_goal.items():
                if slot_type not in self.status[domain]:
                    self.status[domain][slot_type] = {}
                for slot in sub_domain_goal:
                    if slot not in self.status[domain][slot_type]:
                        self.status[domain][slot_type][slot] = {}
                    self.status[domain][slot_type][slot] = {
                        "value": str(sub_domain_goal[slot]),
                        "status": NOT_MENTIONED}

    def get_goal_list(self, data_goal=None):
        goal_list = []
        if data_goal:
            # make sure the order!!!
            for domain, intent, slot, _ in data_goal:
                status = self._get_status(domain, intent, slot)
                value = self.domain_goals[domain][intent][slot]
                goal_list.append([intent, domain, slot, value, status])
            return goal_list
        else:
            for domain, domain_goal in self.domain_goals.items():
                for intent, sub_goal in domain_goal.items():
                    for slot, value in sub_goal.items():
                        status = self._get_status(domain, intent, slot)
                        goal_list.append([intent, domain, slot, value, status])

        return goal_list

    def _get_status(self, domain, intent, slot):
        if domain not in self.status:
            return NOT_MENTIONED
        if intent not in self.status[domain]:
            return NOT_MENTIONED
        if slot not in self.status[domain][intent]:
            return NOT_MENTIONED
        return self.status[domain][intent][slot]["status"]

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain, domain_goal in self.status.items():
            if domain not in self.domain_goals:
                continue
            for slot_type, sub_domain_goal in domain_goal.items():
                if slot_type not in self.domain_goals[domain]:
                    continue
                for slot, status in sub_domain_goal.items():
                    if slot not in self.domain_goals[domain][slot_type]:
                        continue
                    # for strict success, turn this on
                    if status["status"] in [NOT_MENTIONED, CONFLICT]:
                        if status["status"] == CONFLICT and slot in ["arrive by", "leave at"]:
                            continue
                        return False
                    if "?" in status["value"]:
                        return False

        return True

    def slots_results(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        results = {
            'slots_expected': [],
            'slots_fullfilled': [],
            'slots_missing': [],
            'slots_with_status': [],
        }
        for domain, domain_goal in self.status.items():
            if domain not in self.domain_goals:
                continue
            for slot_type, sub_domain_goal in domain_goal.items():
                if slot_type not in self.domain_goals[domain]:
                    continue
                for slot, status in sub_domain_goal.items():
                    if slot not in self.domain_goals[domain][slot_type]:
                        continue

                    results['slots_with_status'].append((domain, slot_type, slot, status))
                    results['slots_expected'].append((domain, slot_type, slot))

                    if status["status"] in [NOT_MENTIONED, CONFLICT]:
                        if status["status"] == CONFLICT and slot in ["arrive by", "leave at"]:
                            results['slots_fullfilled'].append((domain, slot_type, slot))
                            continue
                        else:
                            results['slots_missing'].append((domain, slot_type, slot))
                            continue
                    else:
                        results['slots_fullfilled'].append((domain, slot_type, slot))
                        continue

                    if "?" in status["value"]:
                        results['slots_missing'].append((domain, slot_type, slot))
                    else:
                        results['slots_fullfilled'].append((domain, slot_type, slot))

        return results

    def slots_fullfilled_fraction(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        slots_expected = 0
        slots_fullfilled = 0
        for domain, domain_goal in self.status.items():
            if domain not in self.domain_goals:
                continue
            for slot_type, sub_domain_goal in domain_goal.items():
                if slot_type not in self.domain_goals[domain]:
                    continue
                for slot, status in sub_domain_goal.items():
                    if slot not in self.domain_goals[domain][slot_type]:
                        continue

                    slots_expected += 1

                    if all([
                            status["status"] == FULFILLED,
                            status['value'],
                            status['value'] != "?"]):
                        slots_fullfilled += 1

        return slots_fullfilled / slots_expected

    # TODO change to update()?
    def update_user_goal(self, action, char="usr"):
        action = [
            [intent.lower(), domain.lower(), slot.lower(), value]
            for (intent, domain, slot, value) in action]
        mapped_action = []
        for intent, domain, slot, value in action:
            mapped_slot = slot_name_map.get(domain, {}).get(slot)
            if mapped_slot:
                slot = mapped_slot
            else:
                mapped_slot = slot_name_map.get(slot)
                if mapped_slot:
                    slot = mapped_slot
            i = intent
            if intent == 'inform':
                i = 'info'
            elif intent == 'request':
                i = 'reqt'
            mapped_action.append([i, domain, slot, value])

        # update request and booked
        if char == "usr":
            self._user_action_update(mapped_action)
        elif char == "sys":
            self._system_action_update(mapped_action)
        else:
            print("!!!UNKNOWN CHAR!!!")

        if self.print_details:
            print('\nMAPPED DA:', mapped_action)
            print('STATUS:', self.status)

    def _user_action_update(self, action):
        # no need to update user goal
        for intent, domain, slot, _ in action:
            goal_intent = self._check_slot_and_intent(domain, slot)
            if not goal_intent:
                continue
            # fulfilled by user
            if is_inform(intent):
                self._set_status(goal_intent, domain, slot, FULFILLED)
            # requested by user
            if is_request(intent):
                self._set_status(goal_intent, domain, slot, REQUESTED)

    def _system_action_update(self, action):
        for intent, domain, slot, value in action:
            goal_intent = self._check_slot_and_intent(domain, slot)
            if not goal_intent:
                continue
            # fulfill request by system
            if is_inform(intent) and is_request(goal_intent):
                self._set_status(goal_intent, domain, slot, FULFILLED)
                self._set_goal(goal_intent, domain, slot, value)

            if is_inform(intent) and is_inform(goal_intent):
                # fulfill inform by system
                if value == self.domain_goals[domain][goal_intent][slot]:
                    self._set_status(goal_intent, domain, slot, FULFILLED)
                # conflict system inform
                else:
                    self._set_status(goal_intent, domain, slot, CONFLICT)
            # requested by system
            if is_request(intent) and is_inform(goal_intent):
                self._set_status(goal_intent, domain, slot, REQUESTED)

    def _set_status(self, intent, domain, slot, status):
        self.status[domain][intent][slot]["status"] = status

    def _set_goal(self, intent, domain, slot, value):
        # old_value = self.domain_goals[domain][intent][slot]
        self.domain_goals[domain][intent][slot] = value
        self.status[domain][intent][slot]["value"] = value
        # print(
        #     f"updating user goal {intent}-{domain}-{slot} {old_value}-> {value}")

    def _check_slot_and_intent(self, domain, slot):
        not_found = ""
        if domain not in self.domain_goals:
            return not_found
        for intent in self.domain_goals[domain]:
            if slot in self.domain_goals[domain][intent]:
                return intent
        return not_found


def is_inform(intent):
    if "info" in intent:  # before 'inform'
        return True
    return False


def is_request(intent):
    if "reqt" in intent:  # before 'request'
        return True
    return False


def transform_data_act(data_action):
    action_list = []
    for _, dialog_act in data_action.items():
        for act in dialog_act:
            value = act.get("value", "")
            if not value:
                if "request" in act["intent"]:
                    value = "?"
                else:
                    value = "none"
            action_list.append(
                [act["intent"], act["domain"], act["slot"], value])
    return action_list
