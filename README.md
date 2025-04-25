# ML_Chess_Moves

This project builds on [Chess-Game-using-Pygame](https://github.com/Salow-Studios/Chess-Game-using-Pygame) which provided the original base for the chess game UI.

## Description

This project implements a playable chess game using Pygame, enhanced with a machine learning model that suggests optimal next moves based on board state. The system learns over time from player victories by saving game positions and retraining the model, gradually improving its move suggestions.

## How to Play
This is a machine learning-powered chess game built with Pygame. You play as White, and the computer plays as Black. Here's how to use the program:

Controls:
🖱 Mouse Click: Select and move your pieces.

🔴 RESET Button (Top Right): Restart the game and retrain the AI model using newly saved positions from finished games.

🔤 Press H Key: Get a machine-learning suggested move for the current player.

A green highlight will show which piece the AI recommends moving.

A yellow highlight will show where the AI recommends moving that piece.

The AI learns from its past games using a lightweight set of board evaluation features like material count, king safety, and central control.
