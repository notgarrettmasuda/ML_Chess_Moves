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

    return [material, mobility, opponent_mobility, king_safety, control_center]

# Load data from a CSV file with FEN and Evaluation columns
def load_data(csv_path, limit=100000):
    df = pd.read_csv(csv_path).head(limit)
    df["Evaluation"] = df["Evaluation"].astype(str).str.replace(r"[^\d\-\.]", "", regex=True)
    df["Evaluation"] = pd.to_numeric(df["Evaluation"], errors="coerce")
    df = df.dropna(subset=["Evaluation"])

    X = np.array([extract_features_from_fen(fen) for fen in df["FEN"]])
    y = df["Evaluation"].clip(-1500, 1500) / 1000  # Normalize to [-1.5, 1.5]
    return X, y

# Save a single game state to a CSV for reinforcement
def save_game_state(fen, evaluation, path="learned_games.csv"):
    df = pd.DataFrame([[fen, evaluation]], columns=["FEN", "Evaluation"])
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, mode='w', header=True, index=False)

# Train and save model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=32, random_state=1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model R² score: {score:.2f}")
    joblib.dump(model, "chess_model.joblib")
    print("Model saved as 'chess_model.joblib'")

if __name__ == "__main__":
    X, y = load_data("tactic_evals.csv")
    train_model(X, y)
