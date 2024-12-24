def get_teams(dict):
    """
    This function takes a dictionary as input and returns two dictionaries,
    one for the blue team and one for the red team.
    output_dict contains element like {1:info1, 2:info2, ...}
    """
    blue_team = {}
    red_team = {}
    for key in dict:
        team = key.split("_")[0]
        number = key.split("_")[1]
        if team == "blue":
            blue_team[number] = dict[key]
        else:
            red_team[number] = dict[key]
    return blue_team, red_team

def get_team_members_battle(agents):
    blue_team = []
    red_team = []
    for agent in agents:
        team = agent.split("_")[0]
        number = agent.split("_")[1]
        if team == "blue":
            blue_team.append(agent)
        else:
            red_team.append(agent)
    return blue_team, red_team

def get_team_members_combined_arm(agents):
    blue_melee = []
    blue_ranged = []
    red_melee = []
    red_ranged = []
    for agent in agents:
        name, number = agent.split("_")
        if name == "bluemele":
            blue_melee.append(agent)
        elif name == "blueranged":
            blue_ranged.append(agent)
        elif name == "redmelee":
            red_melee.append(agent)
        else:
            red_ranged.append(agent)
    return blue_melee, blue_ranged, red_melee, red_ranged

def transform_to_onehot(action,num_actions):
    """
    This function takes an action and returns a one-hot encoded action.
    """
    one_hot = [0]*num_actions
    one_hot[action] = 1
    return one_hot
