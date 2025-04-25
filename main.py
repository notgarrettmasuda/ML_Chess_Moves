import pygame
import sys
import os
from joblib import load
from train_from_csv import extract_features_from_fen, save_game_history, load_data, train_model

pygame.init()
model = load("chess_model.joblib")

WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
WHITE, BLACK, BROWN, YELLOW = (255, 255, 255), (0, 0, 0), (139, 69, 19), (255, 255, 0)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game")
font = pygame.font.SysFont("Arial", 32)

suggested_move = None

def init_board():
    global board
    board[1] = [ChessPiece('black', 'pawn', 'images/black_pawn.png') for _ in range(8)]
    board[6] = [ChessPiece('white', 'pawn', 'images/white_pawn.png') for _ in range(8)]
    order = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']
    for i, piece in enumerate(order):
        board[0][i] = ChessPiece('black', piece, f'images/black_{piece}.png')
        board[7][i] = ChessPiece('white', piece, f'images/white_{piece}.png')

class ChessPiece:
    def __init__(self, color, type, image):
        self.color, self.type = color, type
        self.image = pygame.transform.scale(pygame.image.load(image), (SQUARE_SIZE, SQUARE_SIZE))
        self.has_moved = False

def fen_from_board(board, current_player):
    mapping = {'pawn': 'P', 'knight': 'N', 'bishop': 'B', 'rook': 'R', 'queen': 'Q', 'king': 'K'}
    fen_rows = []
    for row in board:
        empty, fen_row = 0, ""
        for piece in row:
            if piece is None:
                empty += 1
            else:
                if empty: fen_row += str(empty)
                empty = 0
                sym = mapping[piece.type]
                fen_row += sym if piece.color == 'white' else sym.lower()
        if empty: fen_row += str(empty)
        fen_rows.append(fen_row)
    return f"{'/'.join(fen_rows)} {'w' if current_player == 'white' else 'b'} KQkq - 0 1"

def draw_board():
    for r in range(8):
        for c in range(8):
            pygame.draw.rect(screen, WHITE if (r + c) % 2 == 0 else BROWN, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    if suggested_move:
        (r1, c1), (r2, c2) = suggested_move
        pygame.draw.rect(screen, YELLOW, (c1 * SQUARE_SIZE, r1 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)
        pygame.draw.rect(screen, YELLOW, (c2 * SQUARE_SIZE, r2 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

def draw_pieces():
    for r in range(8):
        for c in range(8):
            if board[r][c]:
                screen.blit(board[r][c].image, (c * SQUARE_SIZE, r * SQUARE_SIZE))

def draw_reset_button():
    rect = pygame.Rect(WIDTH - 90, 10, 80, 30)
    pygame.draw.rect(screen, (200, 0, 0), rect)
    screen.blit(font.render("RESET", True, WHITE), (rect.x + 10, rect.y + 2))

def get_valid_moves(piece, r, c):
    moves, dirs = [], []
    if piece.type == 'pawn':
        d = -1 if piece.color == 'white' else 1
        if board[r + d][c] is None: moves.append((r + d, c))
        if (piece.color == 'white' and r == 6 or piece.color == 'black' and r == 1) and board[r + 2 * d][c] is None:
            moves.append((r + 2 * d, c))
        for dc in [-1, 1]:
            if 0 <= c + dc < 8 and board[r + d][c + dc] and board[r + d][c + dc].color != piece.color:
                moves.append((r + d, c + dc))
    elif piece.type in ['rook', 'bishop', 'queen']:
        if piece.type == 'rook': dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif piece.type == 'bishop': dirs = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
        else: dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                if board[nr][nc] is None:
                    moves.append((nr, nc))
                elif board[nr][nc].color != piece.color:
                    moves.append((nr, nc))
                    break
                else: break
                nr += dr; nc += dc
    elif piece.type == 'knight':
        for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and (board[nr][nc] is None or board[nr][nc].color != piece.color):
                moves.append((nr, nc))
    elif piece.type == 'king':
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr or dc:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and (board[nr][nc] is None or board[nr][nc].color != piece.color):
                        moves.append((nr, nc))
    return moves

def check_king_alive():
    return any(p for row in board for p in row if p and p.type == 'king' and p.color == 'white') and \
           any(p for row in board for p in row if p and p.type == 'king' and p.color == 'black')

def suggest_move(model, board, current_player):
    best_score, best_move = float('-inf'), None
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece and piece.color == current_player:
                for move in get_valid_moves(piece, r, c):
                    temp, board[move[0]][move[1]] = board[move[0]][move[1]], piece
                    board[r][c] = None
                    score = model.predict([extract_features_from_fen(fen_from_board(board, current_player))])[0]
                    board[r][c], board[move[0]][move[1]] = piece, temp
                    if score > best_score:
                        best_score, best_move = score, ((r, c), move)
    return best_move

def reset_game():
    global board, current_player, selected_piece, selected_pos, model, game_history, game_over, suggested_move
    board = [[None]*8 for _ in range(8)]
    selected_piece = selected_pos = suggested_move = None
    current_player = 'white'
    game_history = []
    init_board()
    try:
        X, y = load_data("learned_games.csv")
        model = train_model(X, y)
    except Exception as e:
        print(f"[WARNING] Could not retrain: {e}")
    game_over = False

def handle_click(pos):
    global selected_piece, selected_pos, current_player, suggested_move
    if WIDTH - 90 <= pos[0] <= WIDTH - 10 and 10 <= pos[1] <= 40:
        reset_game()
        return
    col, row = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE
    if suggested_move and suggested_move[1] == (row, col):
        (r1, c1), (r2, c2) = suggested_move
        board[r2][c2] = board[r1][c1]
        board[r1][c1] = None
        game_history.append((fen_from_board(board, current_player), 1 if current_player == 'white' else -1))
        current_player = 'black' if current_player == 'white' else 'white'
        suggested_move = None
        return
    if selected_piece:
        if (row, col) in get_valid_moves(selected_piece, *selected_pos):
            board[row][col] = selected_piece
            board[selected_pos[0]][selected_pos[1]] = None
            selected_piece.has_moved = True
            if selected_piece.type == 'pawn' and (row == 0 or row == 7):
                board[row][col] = ChessPiece(selected_piece.color, 'queen', f'images/{selected_piece.color}_queen.png')
            game_history.append((fen_from_board(board, current_player), 1 if current_player == 'white' else -1))
            current_player = 'black' if current_player == 'white' else 'white'
        selected_piece = selected_pos = None
    elif board[row][col] and board[row][col].color == current_player:
        selected_piece, selected_pos = board[row][col], (row, col)
        suggested_move = None

def main():
    global board, current_player, game_over, game_history
    game_over = False
    reset_game()
    clock = pygame.time.Clock()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                handle_click(pygame.mouse.get_pos())
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_h:
                move = suggest_move(model, board, current_player)
                if move:
                    global suggested_move
                    suggested_move = move

        if not game_over and not check_king_alive():
            print(f"[INFO] Game Over. Saving {len(game_history)} positions")
            save_game_history(game_history)
            game_over = True

        draw_board(); draw_pieces(); draw_reset_button(); pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()