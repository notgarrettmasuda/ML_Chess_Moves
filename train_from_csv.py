# Chess ML Training Script: PGN -> Model
import chess
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Assign numeric values to pieces for material evaluation
def piece_value(symbol):
    values = {
        'p': 1, 'n': 3, 'b': 3.5, 'r': 5, 'q': 9.5, 'k': 20,
        'P': -1, 'N': -3, 'B': -3.5, 'R': -5, 'Q': -9.5, 'K': -20
    }
    return values.get(symbol, 0)

# Extract enhanced features from FEN
def extract_features_from_fen(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    material = sum(piece_value(piece.symbol()) for piece in piece_map.values())
    mobility = len(list(board.legal_moves))

    board.push(chess.Move.null())  # Switch turn to get opponent's mobility
    opponent_mobility = len(list(board.legal_moves))
    board.pop()

    king_safety = int(board.is_check())
    control_center = sum(
        board.is_attacked_by(board.turn, sq)
        for sq in [chess.D4, chess.E4, chess.D5, chess.E5]
    )

    white_king_pos = board.king(chess.WHITE)
    black_king_pos = board.king(chess.BLACK)
    king_distance = 0
    if white_king_pos is not None and black_king_pos is not None:
        king_distance = abs(chess.square_file(white_king_pos) - chess.square_file(black_king_pos)) + \
                        abs(chess.square_rank(white_king_pos) - chess.square_rank(black_king_pos))

    return [material, mobility, opponent_mobility, king_safety, control_center, king_distance]

# Validate FEN string
def is_valid_fen(fen):
    try:
        chess.Board(fen)
        return True
    except:
        return False

# Load data from a CSV file with FEN and Evaluation columns
def load_data(csv_path, limit=100000):
    df = pd.read_csv(csv_path).head(limit)
    if "Evaluation" not in df.columns or "FEN" not in df.columns:
        raise KeyError("[ERROR] Missing required columns in CSV. Found: " + str(df.columns.tolist()))

    df = df[df["FEN"].apply(is_valid_fen)]
    df["Evaluation"] = df["Evaluation"].astype(str).str.replace(r"[^\d\-\.]", "", regex=True)
    df["Evaluation"] = pd.to_numeric(df["Evaluation"], errors="coerce")
    df = df.dropna(subset=["Evaluation"])

    if df.empty:
        raise ValueError("No valid rows to train on after filtering.")

    X = np.array([extract_features_from_fen(fen) for fen in df["FEN"]])
    y = df["Evaluation"].clip(-1500, 1500) / 1000
    return X, y

# Save multiple game states from a list of (fen, evaluation) pairs
def save_game_history(fen_eval_list, path="learned_games.csv"):
    entries = [(fen, evaluation) for fen, evaluation in fen_eval_list if is_valid_fen(fen)]
    if not entries:
        print("[WARNING] No valid FENs to save from game history.")
        return

    new_data = pd.DataFrame(entries, columns=["FEN", "Evaluation"])
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            if "FEN" not in existing.columns or "Evaluation" not in existing.columns:
                print("[WARNING] Existing CSV is corrupted or missing headers. Rewriting file.")
                new_data.to_csv(path, mode='w', header=True, index=False)
                return
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined.drop_duplicates(subset=["FEN"], inplace=True)
            combined.to_csv(path, index=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print("[WARNING] CSV file is empty or malformed. Rewriting file.")
            new_data.to_csv(path, mode='w', header=True, index=False)
    else:
        new_data.to_csv(path, mode='w', header=True, index=False)

# Train and save model
def train_model(X, y):
    if len(X) < 2:
        raise ValueError("[ERROR] Not enough data to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=32, random_state=1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model R² score: {score:.2f}")
    joblib.dump(model, "chess_model.joblib")
    print("Model saved as 'chess_model.joblib'")
    return model
