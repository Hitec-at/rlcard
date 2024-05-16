import numpy as np

from rlcard.games.base import Card

def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

def get_device():
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("--> Running on the GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device    

def init_standard_deck():
    ''' Initialize a standard deck of 52 cards

    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res

def init_54_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res

def rank2int(rank):
    ''' Get the coresponding number of a rank.

    Args:
        rank(str): rank stored in Card object

    Returns:
        (int): the number corresponding to the rank

    Note:
        1. If the input rank is an empty string, the function will return -1.
        2. If the input rank is not valid, the function will return None.
    '''
    if rank == '':
        return -1
    elif rank.isdigit():
        if int(rank) >= 2 and int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None

def elegent_form(card):
    ''' Get a elegent form of a card string

    Args:
        card (string): A card string

    Returns:
        elegent_card (string): A nice form of card
    '''
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣','s': '♠', 'h': '♥', 'd': '♦', 'c': '♣' }
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank

def print_card(cards):
    ''' Nicely print a card or list of cards

    Args:
        card (string or list): The card(s) to be printed
    '''
    if cards is None:
        cards = [None]
    if isinstance(cards, str):
        cards = [cards]

    lines = [[] for _ in range(9)]

    for card in cards:
        if card is None:
            lines[0].append('┌─────────┐')
            lines[1].append('│░░░░░░░░░│')
            lines[2].append('│░░░░░░░░░│')
            lines[3].append('│░░░░░░░░░│')
            lines[4].append('│░░░░░░░░░│')
            lines[5].append('│░░░░░░░░░│')
            lines[6].append('│░░░░░░░░░│')
            lines[7].append('│░░░░░░░░░│')
            lines[8].append('└─────────┘')
        else:
            if isinstance(card, Card):
                elegent_card = elegent_form(card.suit + card.rank)
            else:
                elegent_card = elegent_form(card)
            suit = elegent_card[0]
            rank = elegent_card[1]
            if len(elegent_card) == 3:
                space = elegent_card[2]
            else:
                space = ' '

            lines[0].append('┌─────────┐')
            lines[1].append('│{}{}       │'.format(rank, space))
            lines[2].append('│         │')
            lines[3].append('│         │')
            lines[4].append('│    {}    │'.format(suit))
            lines[5].append('│         │')
            lines[6].append('│         │')
            lines[7].append('│       {}{}│'.format(space, rank))
            lines[8].append('└─────────┘')

    for line in lines:
        print ('   '.join(line))

def reorganize(trajectories, payoffs):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player])-2, 2):
            if i ==len(trajectories[player])-3:
                reward = payoffs[player]
                done =True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i+3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories

def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector

    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.

    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs

def tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    return payoffs

def plot_curve(csv_path, save_path, algorithm):
    ''' Read data from csv file and plot the results
    '''
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['episode']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

def print_doudizhu_state(state, action_chosen, mask_hand_players: list=[], debug=False):
    ''' (Only for usages of game Doudizhu) Given player state and chosen action and print

    Args:
        state (dict): A dictionary of the raw state
        action_chosen (str): An index pointed to the chosen action. For more infomation, please refer to `ID_2_ACTION`
    '''                     
    print(f'\n============= Game Round {len(state["trace"]) + 1} =============\n')
    
    hand_infos = []
    for player_id, hand_info in enumerate(state['players_current_hands']):
        hand_infos.append({
            'id': hand_info['ID'],
            'role': hand_info['Role'],
            'hand': hand_info['Hand'] if player_id in mask_hand_players else '*' * len(hand_info['Hand']),
            'action': action_chosen if hand_info['ID'] == state['self'] else None
        })
        
    # format players info
    players_info = []
    for hand_info in hand_infos:
        if state['self'] == hand_info['id']:
            player_info = f'-> Player {hand_info["id"]} ({hand_info["role"]}): '
        else:
            player_info = f'   Player {hand_info["id"]} ({hand_info["role"]}): '
        players_info.append(player_info) 
    # entitle
    title = '   Player Info'
    players_info.insert(0, title)
    
    # format last round actions
    last_round_actions = [f'{action}  ' for action in state['played_cards']] 
    if len(state['trace']) > 3: # if actions are more than 3, then the last 3 actions are shown
        for i in range(len(state['trace']) - 3, len(state['trace'])):
            last_round_actions[state['trace'][i][0]] = f'{state["trace"][i][1]}  '
    # entitle
    title = 'Last Round Action   '
    last_round_actions.insert(0, title)
    
    hands = []
    for hand_info in hand_infos:
        hand = f'{hand_info["hand"]}  '
        hands.append(hand) 
    # add current round action to that player
    idx = 0
    for hand, hand_info in zip(hands, hand_infos):
        if state['self'] == hand_info['id']:
            hand += f'-> {hand_info["action"]}  '
            hands[idx] = hand
        idx += 1
    # entitle
    title = 'Hand'
    hands.insert(0, title) 
        
    _align_string_list(players_info)
    _align_string_list(last_round_actions)
    _align_string_list(hands)
    idx = 0
    for player_info, last_round_action, hand in zip(players_info, last_round_actions, hands):
        if idx == 1:
            print('  -----------------------------------------------------------')
        print(player_info, last_round_action, hand)
        idx += 1
    
    if debug:    
        print(f'\n============= Raw State =============\n')
        print(state)
        

def _align_string_list(l):
    max_len = 0
    for s in l:
        if max_len < len(s):
            max_len = len(s)
            
    for i, s in enumerate(l):
        gap = max_len - len(s)
        if gap > 0:
            l[i] += ' ' * gap