from utils import transform_to_onehot
from collections import deque
import random

class LastActionbuffer:
    def __init__(self,actions_melee,actions_ranged,num_melee,num_ranged):
        self.actions_melee = actions_melee
        self.actions_ranged = actions_ranged
        self.redmelee_buffer = {i:[0]*actions_melee for i in range(num_melee)}
        self.redranged_buffer = {i:[0]*actions_ranged for i in range(num_ranged)}
        self.bluemelee_buffer = {i:[0]*actions_melee for i in range(num_melee)}
        self.blueranged_buffer = {i:[0]*actions_ranged for i in range(num_ranged)}

    def get_mean_action(self):
        mean_redmelee_action = [0]*self.actions_melee
        mean_redranged_action = [0]*self.actions_ranged
        mean_bluemelee_action = [0]*self.actions_melee
        mean_blueranged_action = [0]*self.actions_ranged
        for key in self.redmelee_buffer:
            mean_redmelee_action = [x+y for x,y in zip(mean_redmelee_action,self.redmelee_buffer[key])]
        mean_redmelee_action = list(map(lambda x: x / len(self.redmelee_buffer), mean_redmelee_action))
        for key in self.redranged_buffer:
            mean_redranged_action = [x+y for x,y in zip(mean_redranged_action,self.redranged_buffer[key])]
        mean_redranged_action = list(map(lambda x: x / len(self.redranged_buffer), mean_redranged_action))
        for key in self.bluemelee_buffer:
            mean_bluemelee_action = [x+y for x,y in zip(mean_bluemelee_action,self.bluemelee_buffer[key])]
        mean_bluemelee_action = list(map(lambda x: x / len(self.bluemelee_buffer), mean_bluemelee_action))
        for key in self.blueranged_buffer:
            mean_blueranged_action = [x+y for x,y in zip(mean_blueranged_action,self.blueranged_buffer[key])]
        mean_blueranged_action = list(map(lambda x: x / len(self.blueranged_buffer), mean_blueranged_action))
        return mean_bluemelee_action, mean_blueranged_action, mean_redmelee_action, mean_redranged_action

    def flush_buffer(self,actions):
        self.redmelee_buffer = {}
        self.redranged_buffer = {}
        self.bluemelee_buffer = {}
        self.blueranged_buffer = {}
        for key in actions:
            name,number = key.split("_")[0],key.split("_")[1]
            if name=="redmelee":
                self.redmelee_buffer[number] = transform_to_onehot(actions[key],num_actions=self.actions_melee)
            elif name=="redranged":
                self.redranged_buffer[number] = transform_to_onehot(actions[key],num_actions=self.actions_ranged)
            elif name=="bluemele":
                self.bluemelee_buffer[number] = transform_to_onehot(actions[key],num_actions=self.actions_melee)
            else:
                self.blueranged_buffer[number] = transform_to_onehot(actions[key],num_actions=self.actions_ranged)

class Buffer:
    def __init__(self):
        self.bluemelee_buffer = deque(maxlen=10000)
        self.blueranged_buffer = deque(maxlen=10000)
        self.redmelee_buffer = deque(maxlen=10000)
        self.redranged_buffer = deque(maxlen=10000)

    def push(self, data,name):
        if name == "bluemele":
            self.bluemelee_buffer.append(data)
        elif name == "blueranged":
            self.blueranged_buffer.append(data)
        elif name == "redmelee":
            self.redmelee_buffer.append(data)
        else:
            self.redranged_buffer.append(data)

    def process_data(self, data):
        pass

    def sample(self, batch_size,team):
        if team == "blue":
            return random.sample(self.bluemelee_buffer, batch_size),random.sample(self.blueranged_buffer, batch_size)
        else:
            return random.sample(self.redmelee_buffer, batch_size),random.sample(self.redranged_buffer, batch_size)