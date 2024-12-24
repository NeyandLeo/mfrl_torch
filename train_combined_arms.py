from magent2.environments import combined_arms_v6
from buffer_combined_arms import LastActionbuffer, Buffer
from utils import get_team_members_combined_arm
from models import choose_model
config = {
    "episodes": 2000,
    "self_melee":"MFQ",
    "self_ranged":"MFQ",
    "oppo_melee":"IL",
    "oppo_ranged":"IL",
}
# 绿色代表我方（屏幕右侧），红色代表敌方
parallel_env = combined_arms_v6.parallel_env(render_mode='human',max_cycles=3000,map_size=45,extra_features=False)
buffer = Buffer()
self_melee_model = choose_model(config["self_melee"],input_channels=9,num_actions=9)
self_ranged_model = choose_model(config["self_ranged"],input_channels=9,num_actions=25)
oppo_melee_model = choose_model(config["oppo_melee"],input_channels=9,num_actions=9)
oppo_ranged_model = choose_model(config["oppo_ranged"],input_channels=9,num_actions=25)
steps = 0

for i in range(0, config["episodes"]):
    observations = parallel_env.reset(seed=42)
    action_buffer = LastActionbuffer(actions_melee=9,actions_ranged=25,num_melee=45,num_ranged=36)
    while parallel_env.agents:
        steps += 1
        old_observations = observations
        agent_list = parallel_env.agents
        blue_melee, blue_ranged, red_melee, red_ranged = get_team_members_combined_arm(agent_list) #将四个team打包成两个list分开，方便后续处理
        mean_bluemelee_action, mean_blueranged_action, mean_redmelee_action, mean_redranged_action = action_buffer.get_mean_action()#获取上一轮的平均动作
        #=================获取动作=================
        actions = {}
        for member in blue_melee:
            idx = member.split("_")[1]
            if config["self_melee"] == "MFQ":
                actions[member] = self_melee_model.get_action(old_observations[member], idx, mean_bluemelee_action)
            else:
                actions[member] = self_melee_model.get_action(old_observations[member], idx)
        for member in blue_ranged:
            idx = member.split("_")[1]
            if config["self_ranged"] == "MFQ":
                actions[member] = self_ranged_model.get_action(old_observations[member], idx, mean_blueranged_action)
            else:
                actions[member] = self_ranged_model.get_action(old_observations[member], idx)
        for member in red_melee:
            idx = member.split("_")[1]
            if config["oppo_melee"] == "MFQ":
                actions[member] = oppo_melee_model.get_action(old_observations[member], idx, mean_redmelee_action)
            else:
                actions[member] = oppo_melee_model.get_action(old_observations[member], idx)
        for member in red_ranged:
            idx = member.split("_")[1]
            if config["oppo_ranged"] == "MFQ":
                actions[member] = oppo_ranged_model.get_action(old_observations[member], idx, mean_redranged_action)
            else:
                actions[member] = oppo_ranged_model.get_action(old_observations[member], idx)

        #=================执行动作=================
        action_buffer.flush_buffer(actions) #将动作存入buffer,为下一轮计算平均动作做准备
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)#执行动作，获取新的观测值，奖励，终止信号等
        #=================存储经验================
        for agent in agent_list:
            name,number = agent.split("_")[0],agent.split("_")[1]
            old_obs,obs,reward,done = old_observations[agent],observations[agent], rewards[agent], terminations[agent]
            if name=="bluemele":
                buffer.push((old_obs,obs,mean_bluemelee_action,actions[agent],reward,done,number),name)
            elif name=="blueranged":
                buffer.push((old_obs,obs,mean_blueranged_action,actions[agent],reward,done,number),name)
            elif name=="redmelee":
                buffer.push((old_obs,obs,mean_redmelee_action,actions[agent],reward,done,number),name)
            else:
                buffer.push((old_obs,obs,mean_redranged_action,actions[agent],reward,done,number),name)
        #=================训练=================
        if len(buffer.bluemelee_buffer) > 32 and len(buffer.blueranged_buffer) > 32:
            melee_batch,ranged_batch = buffer.sample(32,team="blue")
            blue_melee_loss=self_melee_model.train(melee_batch)
            blue_ranged_loss=self_ranged_model.train(ranged_batch)

        if len(buffer.redmelee_buffer) > 32 and len(buffer.redranged_buffer) > 32:
            melee_batch,ranged_batch = buffer.sample(32,team="red")
            red_melee_loss=oppo_melee_model.train(melee_batch)
            red_ranged_loss=oppo_ranged_model.train(ranged_batch)

            #=================更新target=================
        if steps % 10 == 0:
            self_melee_model.update_target()
            self_ranged_model.update_target()
            oppo_melee_model.update_target()
            oppo_ranged_model.update_target()
        if steps% 100 == 0:
            print("Episode: {}, Step: {}, Blue Melee Loss: {}, Blue Ranged Loss: {}, Red Melee Loss: {}, Red Ranged Loss: {}".format(i,steps,blue_melee_loss,blue_ranged_loss,red_melee_loss,red_ranged_loss))

parallel_env.close()
