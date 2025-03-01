import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy

# Part 1: FEN Parsing and Legal Move Generation
def parse_fen(fen):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    return board, legal_moves

def board_to_tensor(board):
    tensor = torch.zeros(18, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece.piece_type - 1 + (6 if piece.color else 0)
            rank, file = divmod(square, 8)
            tensor[channel, 7 - rank, file] = 1
    tensor[12, 0, 0] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[13, 0, 0] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[14, 0, 0] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[15, 0, 0] = board.has_queenside_castling_rights(chess.BLACK)
    if board.ep_square:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        tensor[16, 7 - ep_rank, ep_file] = 1
    tensor[17, 0, 0] = 1 if board.turn else 0
    return tensor

# Part 2: Neural Network and Reinforcement Learning
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(18, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.resblocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) for _ in range(5)])
        self.policy_head = nn.Linear(256 * 8 * 8, 8 * 8 * 73)
        self.value_head = nn.Linear(256 * 8 * 8, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = x + block(x)
        x = x.view(-1, 256 * 8 * 8)
        policy = self.policy_head(x).view(-1, 8, 8, 73)
        value = torch.tanh(self.value_head(x))
        return policy, value

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb_score())

    def ucb_score(self, c_puct=1.0):
        if self.visit_count == 0:
            return float('inf')
        return (self.total_value / self.visit_count) + c_puct * self.prior * np.sqrt(self.parent.visit_count) / (self.visit_count + 1)

class MCTS:
    def __init__(self, model, simulations=100):
        self.model = model
        self.simulations = simulations

    def search(self, board):
        root = MCTSNode(copy.deepcopy(board))
        for _ in range(self.simulations):
            node = root
            while node.children:
                node = node.select_child()
            if node.board.is_game_over():
                result = node.board.result()
                value = 1 if result == '1-0' else -1 if result == '0-1' else 0
            else:
                policy, value = self.model(board_to_tensor(node.board).unsqueeze(0))
                policy = torch.softmax(policy.view(-1), dim=0)
                legal_moves = list(node.board.legal_moves)
                for move in legal_moves:
                    child_board = copy.deepcopy(node.board)
                    child_board.push(move)
                    prior = policy[self.move_to_index(move, node.board)].item()
                    node.children.append(MCTSNode(child_board, node, prior))
                value = value.item()
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                node = node.parent
                value = -value
        return root

    def move_to_index(self, move, board):
        from_sq = move.from_square
        to_sq = move.to_square
        dx = chess.square_file(to_sq) - chess.square_file(from_sq)
        dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)
        # Simplified mapping (for demonstration)
        return hash(move) % (8*8*73)

class SelfPlay:
    def __init__(self, model, games=10*(10**10), sims=70):
        self.model = model
        self.games = games
        self.sims = sims
        self.memory = deque(maxlen=10*(10**5))

    def generate_game(self, game_number):
        board = chess.Board()
        mcts = MCTS(self.model, self.sims)
        move_count = 0
        while not board.is_game_over():
            root = mcts.search(board)
            total_visits = sum(c.visit_count for c in root.children)
            probs = {c.board.peek(): c.visit_count / total_visits for c in root.children}
            self.memory.append((board_to_tensor(board), probs, board.turn))
            move = max(probs, key=probs.get)
            board.push(move)
            move_count += 1
            print(f"Game {game_number}, Move {move_count}: {move.uci()}")
        result = 1 if board.result() == '1-0' else -1 if board.result() == '0-1' else 0
        return [(s, p, result if t == chess.WHITE else -result) for (s, p, t) in self.memory]

    def train(self, epochs=3, batch_size=32):
        optimizer = optim.Adam(self.model.parameters())
        if not self.memory:
            return
        states, policies, values = zip(*self.memory)
        dataset = list(zip(states, policies, values))
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            batch = random.sample(dataset, min(batch_size, len(dataset)))
            state_batch = torch.stack([x[0] for x in batch])
            policy_batch = [x[1] for x in batch]
            value_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32)
            optimizer.zero_grad()
            policy_pred, value_pred = self.model(state_batch)
            policy_loss = 0.0
            mcts = MCTS(self.model)
            for i, p in enumerate(policy_batch):
                legal_indices = [mcts.move_to_index(m, batch[i][0]) for m in p.keys()]
                log_probs = torch.log_softmax(policy_pred[i].view(-1), dim=0)
                policy_loss += -torch.sum(torch.tensor(list(p.values())) * log_probs[legal_indices])
            value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), value_batch)
            total_loss = policy_loss + value_loss
            total_loss.backward()
            optimizer.step()
            print(f"Batch {epoch + 1}: Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Total Loss: {total_loss.item()}")

def get_top_moves(fen, model, top_n=3):
    board = chess.Board(fen)
    tensor = board_to_tensor(board).unsqueeze(0)
    policy, _ = model(tensor)
    policy = torch.softmax(policy.view(-1), dim=0)
    legal_moves = list(board.legal_moves)
    move_probs = [(move, policy[MCTS(model).move_to_index(move, board)].item()) for move in legal_moves]
    move_probs.sort(key=lambda x: x[1], reverse=True)
    return [move.uci() for move, _ in move_probs[:top_n]]

if __name__ == "__main__":
    # Part 1 Example
    fen = "2q3k1/1Q5p/2p3p1/2Pp4/3P4/1R3pPP/PP3P1K/4r3 b - - 2 42"
    board, legal_moves = parse_fen(fen)
    print(f"Legal Moves: {[move.uci() for move in legal_moves[:5]]}...")

    # Part 2 Training
    model = ChessNet()
    self_play = SelfPlay(model, games=10*(10**4), sims=70)
    for game in range(self_play.games):
        print(f"Generating game {game + 1}/{self_play.games}")
        self_play.generate_game(game + 1)
        self_play.train()
        torch.save(model.state_dict(), "chess_model.pth")

    # Part 3 Inference
    model.load_state_dict(torch.load("chess_model.pth"))
    top_moves = get_top_moves(fen, model)
    print(f"Top 3 Moves: {top_moves}")