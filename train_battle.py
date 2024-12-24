from magent2.environments import battle_v4
from buffer_battle import LastActionbuffer, Buffer
from utils import get_team_members_battle
from models import choose_model
config = {
    "episodes": 2000,
    "self":"MFQ",
    "oppo":"IL",
}
# 蓝色代表我方（屏幕右侧），红色代表敌方
parallel_env = battle_v4.parallel_env(render_mode='human',max_cycles=3000,map_size=45,extra_features=False)
buffer = Buffer()
self_model = choose_model(config["self"],input_channels=5,num_actions=21)
oppo_model = choose_model(config["oppo"],input_channels=5,num_actions=21)
steps = 0

for i in range(0, config["episodes"]):
    observations = parallel_env.reset(seed=42)
    action_buffer = LastActionbuffer(actions=21,num_agents=81)
    while parallel_env.agents:
        steps += 1
        old_observations = observations
        agent_list = parallel_env.agents
        blue_team,red_team = get_team_members_battle(agent_list) #将两个team打包成两个list分开，方便后续处理
        mean_blue_action, mean_red_action = action_buffer.get_mean_action()#获取上一轮的平均动作
        #=================获取动作=================
        actions = {}
        for member in blue_team:
            idx = member.split("_")[1]
            if config["self"] == "MFQ":
                actions[member] = self_model.get_action(old_observations[member], idx, mean_blue_action)
            else:
                actions[member] = self_model.get_action(old_observations[member], idx)
        for member in red_team:
            idx = member.split("_")[1]
            if config["oppo"] == "MFQ":
                actions[member] = oppo_model.get_action(old_observations[member], idx, mean_blue_action)
            else:
                actions[member] = oppo_model.get_action(old_observations[member], idx)
        #=================执行动作=================
        action_buffer.flush_buffer(actions) #将动作存入buffer,为下一轮计算平均动作做准备
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)#执行动作，获取新的观测值，奖励，终止信号等
        #=================存储经验================
        for agent in agent_list:
            team,number = agent.split("_")[0],agent.split("_")[1]
            old_obs,obs,reward,done = old_observations[agent],observations[agent], rewards[agent], terminations[agent]
            if team == "blue":
                buffer.push((old_obs,obs,mean_blue_action,actions[agent],reward,done,number),team)
            else:
                buffer.push((old_obs,obs,mean_red_action,actions[agent],reward,done,number),team)
        #=================训练=================
        if len(buffer.blue_buffer) > 32:
            batch = buffer.sample(32,team="blue")
            blue_team_loss=self_model.train(batch)
        if len(buffer.red_buffer) > 32:
            batch = buffer.sample(32,team="red")
            red_team_loss=oppo_model.train(batch)
            #=================更新target=================
        if steps % 10 == 0:
            self_model.update_target()
            oppo_model.update_target()
        if steps% 100 == 0:
            print(f"Step:{steps},Blue Team Loss:{blue_team_loss},Red Team Loss:{red_team_loss}")

parallel_env.close()
