import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class HeuristicAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.opponent = PLAYER2 if player == PLAYER1 else PLAYER1

    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        
        if current_pieces < NUM_PIECES:
            # placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def evaluate_board(self, board):
        """Evaluate the board state."""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr, dc in directions:
                    player_count = 0
                    opponent_count = 0
                    for i in range(3):
                        nr, nc = _torus(r + i*dr, c + i*dc)
                        if board[nr][nc] == self.player:
                            player_count += 1
                        elif board[nr][nc] == self.opponent:
                            opponent_count += 1
                    
                    if player_count == 3:
                        score += 100
                    elif player_count == 2 and opponent_count == 0:
                        score += 10
                    elif opponent_count == 3:
                        score -= 100
                    elif opponent_count == 2 and player_count == 0:
                        score -= 10

        return score

    def make_move(self, game, move):
        """Apply the move to a copy of the game and return the new game state."""
        new_game = Game.from_dict(game.to_dict())
        if len(move) == 2:
            new_game.place_checker(*move)
        else:
            new_game.move_checker(*move)
        return new_game

    def get_best_move(self, game):
        """Returns the best move based on the heuristic evaluation."""
        possible_moves = self.get_possible_moves(game)
        best_score = float('-inf')
        best_move = None

        for move in possible_moves:
            new_game = self.make_move(game, move)
            score = self.evaluate_board(new_game.board)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move or random.choice(possible_moves)  # Fallback to random if no best move
