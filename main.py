from magent2.environments import battle_v4
from pettingzoo.utils import random_demo
from buffer import LastActionbuffer, Buffer
from models.mf import MFModel
from utils import get_team_members

config = {
    "episodes": 2000,
}

parallel_env = battle_v4.parallel_env(render_mode='human')
buffer = Buffer()
mfmodel = MFModel()
oppo_model = MFModel()
steps = 0

for i in range(0, config["episodes"]):
    observations = parallel_env.reset(seed=42)
    action_buffer = LastActionbuffer(actions=21,num_agents=81)
    while parallel_env.agents:
        steps += 1
        old_observations = observations
        agent_list = parallel_env.agents
        blue_team,red_team = get_team_members(agent_list) #将两个team打包成两个list分开，方便后续处理
        mean_blue_action, mean_red_action = action_buffer.get_mean_action()#获取上一轮的平均动作
        actions = {}
        for member in blue_team:
            idx = member.split("_")[1]
            actions[member] = mfmodel.get_action(old_observations[member],idx,mean_blue_action)
        for member in red_team:
            idx = member.split("_")[1]
            actions[member] = oppo_model.get_action(old_observations[member],idx,mean_red_action)
        # this is where you would insert your policy
        #actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        action_buffer.flush_buffer(actions)
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        for agent in agent_list:
            team,number = agent.split("_")[0],agent.split("_")[1]
            old_obs,obs,reward,done = old_observations[agent],observations[agent], rewards[agent], terminations[agent]
            if team == "blue":
                buffer.push((old_obs,obs,mean_blue_action,actions[agent],reward,done,number),team)
            else:
                buffer.push((old_obs,obs,mean_red_action,actions[agent],reward,done,number),team)
        if len(buffer.blue_buffer) > 32:
            batch = buffer.sample(32,team="blue")
            loss=mfmodel.train(batch)
            print(f"blueteam---steps: {steps}, loss: {loss}")
        if len(buffer.red_buffer) > 32:
            batch = buffer.sample(32,team="red")
            loss=oppo_model.train(batch)
            print(f"redteam---steps: {steps}, loss: {loss}")
        if steps % 10 == 0:
            mfmodel.update_target()
            oppo_model.update_target()


parallel_env.close()