''' A toy example of playing against pretrianed AI on Doudizhu
'''
from rlcard.agents import RandomAgent

import rlcard
from rlcard import models
# from rlcard.agents import DoudizhuHumanAgent as HumanAgent
from rlcard.utils import print_card

# Make environment
env = rlcard.make('doudizhu')

# human_agent = HumanAgent(env.num_actions)
random_agent1 = RandomAgent(num_actions=env.num_actions)
random_agent2 = RandomAgent(num_actions=env.num_actions)
random_agent3 = RandomAgent(num_actions=env.num_actions)

env.set_agents([random_agent1, random_agent2, random_agent3])

# my_player_id = 0


while (True):
    print(">> A New doudizhu Game!")

    trajectories, payoffs = env.run(is_training=False)

    print('\n===============     Result     ===============\n')
    for i in range(env.num_players):
        print('Player {} gets {} points'.format(i, payoffs[i]))

    input("\n\nPress any key to continue...\n\n")
