import numpy as np
import tensorflow as tf

def heuristic_policy(env):
    """
    Heuristic policy for the Snake game.
    The policy is based on the distance to the fruit and avoids walls and the snake's body.
    The snake will always try to move towards the fruit while avoiding obstacles.
    If there are no valid moves, a random action is chosen.
    Args:
        env: The Snake environment.
    Returns:
        A tensor of shape (n_boards, 1) containing the chosen actions for each board.
    """

    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ACTIONS = [UP, RIGHT, DOWN, LEFT]

    DIRS = {
        UP: (1, 0),
        RIGHT: (0, 1),
        DOWN: (-1, 0),
        LEFT: (0, -1)
    }

    boards = env.boards
    n_boards, board_size, _ = boards.shape
    actions = []

    for i in range(n_boards):
        board = boards[i]
        head = tuple(map(int, np.argwhere(board == env.HEAD)[0]))
        fruit = tuple(map(int, np.argwhere(board == env.FRUIT)[0]))
        candidates = []
        for action in ACTIONS:
            dy, dx = DIRS[action]
            next_pos = (head[0] + dy, head[1] + dx)

            if(0 <= next_pos[0] < board_size) and (0 <= next_pos[1] < board_size):
                target_cell = board[next_pos]
                if target_cell != env.WALL and target_cell != env.BODY:
                    distance = abs(next_pos[0] - fruit[0]) + abs(next_pos[1] - fruit[1])
                    candidates.append((action, distance))
        
        if not candidates:
            action = np.random.choice(ACTIONS)
        else:
            candidates.sort(key=lambda x: x[1])
            action = candidates[0][0]
        actions.append(action)
    
    return tf.convert_to_tensor(actions, dtype=tf.int32)[:, None]