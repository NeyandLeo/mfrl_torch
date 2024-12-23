from utils import transform_to_onehot
from collections import deque
import random

class LastActionbuffer:
    def __init__(self,actions,num_agents):
        self.blue_buffer = {i:[0]*actions for i in range(num_agents)}
        self.red_buffer = {i:[0]*actions for i in range(num_agents)}
    def get_last_action(self):
        return self.blue_buffer, self.red_buffer

    def get_mean_action(self):
        mean_blue_action = [0]*21
        mean_red_action = [0]*21
        for key in self.blue_buffer:
            mean_blue_action = [x+y for x,y in zip(mean_blue_action,self.blue_buffer[key])]
        mean_blue_action = list(map(lambda x: x / len(self.blue_buffer), mean_blue_action))
        for key in self.red_buffer:
            mean_red_action = [x+y for x,y in zip(mean_red_action,self.red_buffer[key])]
        mean_red_action = list(map(lambda x: x / len(self.red_buffer), mean_red_action))
        return mean_blue_action, mean_red_action

    def flush_buffer(self,actions):
        self.blue_buffer = {}
        self.red_buffer = {}
        for key in actions:
            team = key.split("_")[0]
            number = key.split("_")[1]
            if team == "blue":
                self.blue_buffer[number] = transform_to_onehot(actions[key])
            else:
                self.red_buffer[number] = transform_to_onehot(actions[key])

class Buffer:
    def __init__(self):
        self.blue_buffer = deque(maxlen=10000)
        self.red_buffer = deque(maxlen=10000)

    def push(self, data,team):
        if team == "blue":
            self.blue_buffer.append(data)
        else:
            self.red_buffer.append(data)

    def process_data(self, data):
        pass

    def sample(self, batch_size,team):
        if team == "blue":
            return random.sample(self.blue_buffer, batch_size)
        else:
            return random.sample(self.red_buffer, batch_size)