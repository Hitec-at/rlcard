import functools
from rlcard.utils.utils import print_card
from rlcard.games.nolimitholdem import Action
from rlcard.utils.utils import print_doudizhu_state
from rlcard.games.doudizhu.utils import ACTION_2_ID, doudizhu_sort_str


class HumanAgent(object):
    ''' A human agent for Doudizhu. It receives user prompt from command line and does the action accordingly.
    '''

    def __init__(self, num_actions, player_id=0):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.player_id = player_id
        self.use_raw = False
        self.num_actions = num_actions

    def step(self, state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        _print_state(state['raw_obs'], self.player_id)
        action = _action_format(input('>> You choose action : '))
        
        while action not in state['raw_obs']['actions']:
            print('Action illegal...')
            action = _action_format(input('>> Re-choose action : '))
        
        print_doudizhu_state(state['raw_obs'], action, mask_hand_players=[self.player_id])
        action_idx = ACTION_2_ID[action]
        return action_idx

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}
    
    def set_player_id(self, id):
        self.player_id = id

def _print_state(state, player_id):
    ''' Print out the state

    Args:
        state (dict): A dictionary of the raw state
        action_record (list): A list of the historical actions
    '''
    print(f'\n>>> You are playing as player {player_id} ({"Landlord" if player_id == state["self"] else "Peasant"}) <<<\n')
    
    if 'trace' in state and len(state['trace']) > 2:
        print('=========== Last Round Actions ===========')
        print(f'Player {state["trace"][-2][0]} ({"Landlord" if state["trace"][-2][0] == state["self"] else "Peasant"}): {state["trace"][-2][1]}')
        print(f'Player {state["trace"][-1][0]} ({"Landlord" if state["trace"][-1][0] == state["self"] else "Peasant"}): {state["trace"][-1][1]}')
        
    print('\n=========== Your Hand ===========')
    print(state['current_hand'])

def _action_format(action: str) -> str:
    if len(action) == 0:
        return action
    if action == 'pass' or action == 'p':
        return 'pass'
    
    action = action.upper()
    temp_action_list = list(action)
    temp_action_list.sort(key=functools.cmp_to_key(doudizhu_sort_str))
    return ''.join(temp_action_list)