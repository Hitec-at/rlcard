''' A toy example of playing against pretrianed AI on Doudizhu
'''
from rlcard.agents import RandomAgent

import rlcard
from rlcard import models
from rlcard.agents import DoudizhuHumanAgent as HumanAgent
from rlcard.utils import print_card

# Make environment
env = rlcard.make('doudizhu')

my_player_id = 0

human_agent = HumanAgent(env.num_actions, player_id=my_player_id)
random_agent1 = RandomAgent(num_actions=env.num_actions, human_player_ids=[my_player_id])
random_agent2 = RandomAgent(num_actions=env.num_actions, human_player_ids=[my_player_id])
# random_agent3 = RandomAgent(num_actions=env.num_actions)

env.set_agents([
    human_agent, 
    random_agent1, 
    random_agent2,
    # random_agent3,
])


while (True):
    print("\n>> A New doudizhu Game!\n")

    trajectories, payoffs = env.run(is_training=False)
    
    print("\n>> Game Over!\n")
    for i in range(env.num_players):
        if i == my_player_id:
            if payoffs[i] > 0:
                print('\n===============     You Win!     ===============\n')
            else:
                print('\n===============     You Lose!     ===============\n')
        print('Player {} gets {} points'.format(i, payoffs[i]))

    input("\n\nPress any key to continue...\n\n")
