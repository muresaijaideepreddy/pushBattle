import numpy as np
import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES

class QLearningAgent:
    def __init__(self, player=PLAYER1, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}

    def get_state_key(self, game):
        return str(game.board.tolist())

    def get_possible_moves(self, game):
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        
        if current_pieces < NUM_PIECES:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def get_best_move(self, game):
        state = self.get_state_key(game)
        if random.random() < self.exploration_rate:
            return random.choice(self.get_possible_moves(game))
        
        if state not in self.q_table:
            return random.choice(self.get_possible_moves(game))
        
        q_values = self.q_table[state]
        best_move = max(q_values, key=q_values.get)
        return list(map(int, best_move.split(',')))

    def update_q_table(self, state, action, next_state, reward):
        if state not in self.q_table:
            self.q_table[state] = {str(move): 0 for move in self.get_possible_moves(Game.from_dict(eval(state)))}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {str(move): 0 for move in self.get_possible_moves(Game.from_dict(eval(next_state)))}
        
        current_q = self.q_table[state][str(action)]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][str(action)] = new_q

    def train(self, num_episodes=1000):
        for _ in range(num_episodes):
            game = Game()
            state = self.get_state_key(game)
            
            while game.check_winner() == EMPTY:
                action = self.get_best_move(game)
                if len(action) == 2:
                    game.place_checker(*action)
                else:
                    game.move_checker(*action)
                
                next_state = self.get_state_key(game)
                reward = 1 if game.check_winner() == self.player else -1 if game.check_winner() != EMPTY else 0
                
                self.update_q_table(state, action, next_state, reward)
                
                state = next_state
                game.current_player *= -1

    def save_q_table(self, filename):
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.load(filename, allow_pickle=True).item()
