import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ILQNet(nn.Module):
    def __init__(self,num_actions=21, hidden_size=256):
        super(ILQNet, self).__init__()

        # 1. obs 先经过两层卷积然后展平
        # 假设输入 obs 的形状为 (B, 5, 13, 13)
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # 卷积后展平会得到 32 * 13 * 13 = 5408 的维度
        self.flatten_size = 32 * 13 * 13

        # 用于卷积输出的全连接
        self.fc_obs = nn.Linear(self.flatten_size, 128)

        # 2. feature (shape = (B,1)) 经过一层全连接
        self.fc_feature = nn.Linear(1, 64)

        # 拼接后 x 的维度 = 128 + 64 = 192

        # 4. y 经过三层全连接输出 Q 值
        # 这里可以根据需要调整隐藏层的大小，这里示例设为 hidden_size
        self.fc1 = nn.Linear(192, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, obs, feature):
        """
        obs: (B, 13, 13, 5)
        feature: (B, 1)
        returns: Q-values, shape (B, 21)
        """
        obs = obs.permute(0, 3, 1, 2)  # (B, 13, 13, 5) => (B, 5, 13, 13)
        # 卷积部分
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x = x.reshape(x.size(0), -1) # 展平 => (B, flatten_size)

        # obs 全连接
        x_obs = F.relu(self.fc_obs(x))

        # feature 全连接
        x_feat = F.relu(self.fc_feature(feature))

        # 连接 obs 和 feature => x
        x_cat = torch.cat([x_obs, x_feat], dim=1)  # (B, 128 + 64 = 192)

        # 三层全连接 -> Q 值
        y = F.relu(self.fc1(x_cat))
        y = F.relu(self.fc2(y))
        q_values = self.fc3(y)

        return q_values

class ILModel(nn.Module):
    def __init__(self,num_actions=21):
        super(ILModel, self).__init__()
        self.qnet = ILQNet(num_actions, hidden_size=256)
        self.target_qnet = ILQNet(num_actions, hidden_size=256)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=1e-3)
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.epsilon = 0.2

    def train(self, batch):
        # 从 batch 中解包数据
        old_obs, obs, mean_action, action, reward, done, idx = zip(*batch)

        # 转成 tensor 并移动到相应设备
        # 下面假设 obs 的形状是 [B, 5, 13, 13]，如果不是需要自行 reshape
        old_obs = torch.tensor(old_obs, dtype=torch.float32, device=self.device)  # (B, 5, 13, 13)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)  # (B, 5, 13, 13)

        # action: (B,) 存储的是选取的离散动作下标
        action = torch.tensor(action, dtype=torch.long, device=self.device)  # (B,)

        # reward, done: (B,)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)  # (B,)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)  # (B,)

        # idx: (B, 1) 或 (B,) ，根据你的网络对 feature 的要求来处理
        # 如果只是一个整数特征，可以把它当做 shape (B, 1)
        idx = [float(x) for x in idx]  # 或者 int(x)
        idx = torch.tensor(idx, dtype=torch.float32, device=self.device).unsqueeze(-1)  # (B,1)

        # =============== 计算当前 Q 值 ===============
        # 使用当前 Q 网络，输入 (old_obs, idx, mean_action)，输出 Q(s, a) shape: (B, num_actions)
        q_values_all = self.qnet(old_obs, idx)  # (B, num_actions)

        # 根据实际选择的 action，取到对应的 Q 值
        # shape: (B,) 每个样本只取对应动作的 Q
        current_q = q_values_all.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # =============== 计算目标 Q 值 ===============
        with torch.no_grad():
            # 使用目标 Q 网络，输入 (obs, idx, mean_action) 得到 next_state 的 Q
            next_q_values_all = self.target_qnet(obs, idx)  # (B, num_actions)
            # 这里是 DQN，取下一个状态里最大的 Q 值
            next_q_max = next_q_values_all.max(dim=1)[0]  # (B,)

            # 如果是 Double DQN，需要用在线网络选动作，再用目标网络估计 Q 值，这里先不展开
            # next_actions = q_values_next_online.max(1)[1] ...

            # 对于非终止状态，目标 Q = r + gamma * max Q(s', a')
            # 对于终止状态，目标 Q = r
            target_q = reward + (1 - done) * self.gamma * next_q_max

        # =============== 计算损失并反向传播 ===============
        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, target_q)  # MSE Loss
        loss.backward()
        self.optimizer.step()

        # 这里可以根据需要进行 target 网络的软更新或者定期硬更新
        # 如果想每个 step 都做软更新： tau=0.01
        # for target_param, local_param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
        #     target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        return loss.item()

    def update_target(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def get_action(self, obs, idx):
        # 输入 obs, idx, mean_action，输出 Q 值最大的动作
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        idx = float(idx)
        idx = torch.tensor([idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, 20)
        else:
            q_values = self.qnet(obs, idx)
            qmax = q_values.max().item()
            possible_actions = [i for i in range(21) if q_values[0][i].item() == qmax]
            return random.choice(possible_actions)