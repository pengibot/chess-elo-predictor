from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, Counter
import chess
import chess.pgn
import pickle
import sys
import pandas as pd


def extract_game_features(game, eval_df):

    # Center squares
    center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}

    """Extract features for a given game, using Stockfish evaluations for blunders and inaccuracies."""
    first_check = True
    first_queen_move = True

    node = game

    material_advantage = 0
    # total_piece_activity = 0
    center_control_white = 0
    center_control_black = 0
    piece_activity_white = defaultdict(int)
    piece_activity_black = defaultdict(int)

    # Initialize evaluation variables

    # Initialization for new features
    features = defaultdict(int)
    first_knight_on_edge = {'white': None, 'black': None}
    moves_before_castling = {'white': None, 'black': None}
    first_blunder = {'white': None, 'black': None}
    first_mistake = {'white': None, 'black': None}
    total_knights_on_edge = {'white': 0, 'black': 0}
    center_control = {'white': 0, 'black': 0}

    # Old Features
    features['white_first_check_at'] = 0
    features['white_first_check_at'] = 0
    features['white_promotion'] = 0
    features['black_promotion'] = 0
    features['white_queen_changed_at'] = 0
    features['black_queen_changed_at'] = 0
    features['white_blunder_count'] = 0
    features['black_blunder_count'] = 0
    features['white_mistake_count'] = 0
    features['black_mistake_count'] = 0
    features['white_inaccuracy_count'] = 0
    features['black_inaccuracy_count'] = 0
    features['white_queen_moved_at'] = 0
    features['black_queen_moved_at'] = 0
    features['white_total_checks'] = 0
    features['black_total_checks'] = 0
    features['white_king_castle'] = 0
    features['black_king_castle'] = 0

    features['white_queen_castle'] = 0
    features['black_queen_castle'] = 0
    features['white_king_castle'] = 0
    features['black_king_castle'] = 0

    previous_evaluation = None

    while node.variations:
        move = node.variation(0).move
        board = node.board()

        # Determine which side is making the move
        current_side = 'white' if board.turn else 'black'

        # 1. **First Knight on Edge** and **Total Knights on Edge**
        if board.piece_type_at(move.from_square) == chess.KNIGHT:
            if chess.square_file(move.to_square) in {0, 7} or chess.square_rank(move.to_square) in {0, 7}:
                total_knights_on_edge[current_side] += 1
                if first_knight_on_edge[current_side] is None:
                    first_knight_on_edge[current_side] = board.fullmove_number

        # 2. **Moves Before Castling**
        uci_repr = move.uci()
        if moves_before_castling[current_side] is None and (
                uci_repr in ['e1g1', 'e1c1', 'e8g8', 'e8c8']
        ):
            moves_before_castling[current_side] = board.fullmove_number

        # 3. **Isolated, Double, and Tripled Pawns**
        pawn_files = [square % 8 for square in board.pieces(chess.PAWN, board.turn)]
        file_counts = Counter(pawn_files)

        features[f'{current_side}_isolated_pawns'] = sum(
            1 for f in file_counts if all(adj not in file_counts for adj in {f - 1, f + 1})
        )
        features[f'{current_side}_double_pawns'] = sum(1 for count in file_counts.values() if count == 2)
        features[f'{current_side}_tripled_pawns'] = sum(1 for count in file_counts.values() if count == 3)

        # 4. **Defending Centre**
        if move.to_square in center_squares:
            center_control[current_side] += 1

        # Piece Activity
        moved_piece = board.piece_type_at(move.from_square)
        if board.turn:  # White's turn
            piece_activity_white[moved_piece] += 1
        else:  # Black's turn
            piece_activity_black[moved_piece] += 1

        captured_piece = board.piece_type_at(move.to_square)

        # Center Control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        if move.to_square in center_squares:
            if board.turn:  # White
                center_control_white += 1
            else:  # Black
                center_control_black += 1

        # Queen moves
        if moved_piece == chess.QUEEN and first_queen_move:
            features[f'{current_side}_queen_moved_at'] = board.fullmove_number
            first_queen_move = False

        if captured_piece == chess.QUEEN:
            features[f'{current_side}_queen_changed_at'] = board.fullmove_number

        # Promotion
        if move.promotion:
            features[f'{current_side}_promotion'] += 1

        # Checks
        if board.is_check():
            features[f'{current_side}_total_checks'] += 1
            if first_check:
                features[f'{current_side}_first_check_at'] = board.fullmove_number
                first_check = False

        # Castling
        uci_repr = move.uci()
        if uci_repr == 'e1g1':
            features['white_king_castle'] = board.fullmove_number
        elif uci_repr == 'e1c1':
            features['white_queen_castle'] = board.fullmove_number
        elif uci_repr == 'e8g8':
            features['black_king_castle'] = board.fullmove_number
        elif uci_repr == 'e8c8':
            features['black_queen_castle'] = board.fullmove_number

        # Material Imbalance
        material_balance = sum(
            [board.piece_map()[sq].piece_type for sq in board.piece_map()]
        )
        material_advantage += material_balance if board.turn else -material_balance

        next_node = node.variation(0)
        currentboard = next_node.board()

        # Blunders, Mistakes, and Inaccuracies
        move_eval = eval_df.loc["evals"]
        if len(move_eval) > 0 and not currentboard.is_checkmate():

            # Full move number (1-based index, so 1 means White's first move)
            fullmove_number = board.fullmove_number

            # Determine actual move count (half-moves or ply)
            if board.turn:  # White's turn
                move_count = 2 * (fullmove_number - 1) + 1
            else:  # Black's turn
                move_count = 2 * fullmove_number

            if (move_count - 1) < len(move_eval):
                current_evaluation = move_eval[move_count - 1]
            else:
                print(f"Error: move_count {move_count} exceeds available evaluations for game: {game}")
                break  # Or handle this case appropriately

            if previous_evaluation is not None:
                eval_diff = abs(current_evaluation - previous_evaluation)

                # Determine which side made the move
                current_side = 'white' if board.turn else 'black'

                # Evaluate the difference in evaluation score to identify mistakes
                if eval_diff >= 200:  # Blunder threshold (e.g., losing 2 pawns)
                    features[f'{current_side}_blunder_count'] += 1
                elif eval_diff >= 100:  # Mistake threshold
                    features[f'{current_side}_mistake_count'] += 1
                elif eval_diff >= 50:  # Inaccuracy threshold
                    features[f'{current_side}_inaccuracy_count'] += 1

                # 5. **Blunders and Mistakes (First Instances)**
                if eval_diff >= 200 and first_blunder[current_side] is None:
                    first_blunder[current_side] = board.fullmove_number
                elif eval_diff >= 100 and first_mistake[current_side] is None:
                    first_mistake[current_side] = board.fullmove_number

            previous_evaluation = current_evaluation

        # Check if the current move results in checkmate
        if board.is_checkmate():
            features['is_checkmate'] = 1
            break  # No need to continue if the game has ended in checkmate

        node = node.variation(0)

    board = node.board()
    # Evaluate game result
    if board.is_checkmate():
        features['is_checkmate'] = 1
    if board.is_stalemate():
        features['is_stalemate'] = 1
    if board.is_insufficient_material():
        features['insufficient_material'] = 1
    if board.can_claim_draw():
        features['can_claim_draw'] = 1
    features['total_moves'] = board.fullmove_number

    # # Pieces at the end of the game
    # piece_placement = board.fen().split()[0]
    # end_pieces = Counter(x for x in piece_placement if x.isalpha())
    #
    # features.update({'end_' + piece: cnt for piece, cnt in end_pieces.items()})

    # Get the final board position in FEN format
    final_fen = board.fen()

    # The first part of the FEN string represents the piece placement on the board
    piece_placement = final_fen.split()[0]

    # Count the number of each type of piece remaining on the board
    remaining_pieces_count = Counter(piece for piece in piece_placement if piece.isalpha())

    # Add the counts of each remaining piece to the features dictionary
    # The keys are formatted as 'end_<piece>', e.g., 'end_r' for black rook
    for piece, count in remaining_pieces_count.items():
        features[f'end_{piece}'] = count

    # Calculate overall piece activity
    features['white_piece_activity'] = sum(piece_activity_white.values())
    features['black_piece_activity'] = sum(piece_activity_black.values())

    # Center control
    features['center_control_white'] = center_control_white
    features['center_control_black'] = center_control_black

    # Material advantage
    features['material_advantage'] = material_advantage

    # Add color-specific features to the feature dictionary
    features.update({
        'white_first_knight_on_edge': first_knight_on_edge['white'] if first_knight_on_edge['white'] is not None else 0,
        'black_first_knight_on_edge': first_knight_on_edge['black'] if first_knight_on_edge['black'] is not None else 0,
        'white_moves_before_castling': moves_before_castling['white'] if moves_before_castling[
                                                                             'white'] is not None else 0,
        'black_moves_before_castling': moves_before_castling['black'] if moves_before_castling[
                                                                             'black'] is not None else 0,
        'white_total_knights_on_edge': total_knights_on_edge['white'],
        'black_total_knights_on_edge': total_knights_on_edge['black'],
        'white_center_control': center_control['white'],
        'black_center_control': center_control['black'],
        'white_first_blunder': first_blunder['white'] if first_blunder['white'] is not None else 0,
        'black_first_blunder': first_blunder['black'] if first_blunder['black'] is not None else 0,
        'white_first_mistake': first_mistake['white'] if first_mistake['white'] is not None else 0,
        'black_first_mistake': first_mistake['black'] if first_mistake['black'] is not None else 0
    })

    return features


def score_features(games, evals, return_names=False):
    game_features = []
    for game, eval_df in zip(games, evals):
        features = extract_game_features(game, eval_df)
        game_features.append(features)

    vec = DictVectorizer()
    x = vec.fit_transform(game_features)
    if return_names:
        return x, vec.get_feature_names_out()
    else:
        return x


def get_games(filename, n_games=sys.maxsize):
    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)
        cnt = 0
        while game and cnt < n_games:
            cnt += 1
            if cnt % 100 == 0:
                print(f"Processed {cnt} Games")
            yield game
            game = chess.pgn.read_game(pgn)


def get_evaluations(filename, n_evals=sys.maxsize):
    with open(filename, 'rb') as file:
        eval_df = pickle.load(file)

        for idx in range(min(n_evals, len(eval_df))):
            yield eval_df.iloc[idx]


def main():

    # Score Feature extraction
    print("Started Extracting Score Features to Pickle File...")

    input_directory = Path("../filter-games/Data/TrainingData")
    pickls_directory = Path("Data/Pickls")
    input_filename = r"combined_games.pgn"
    evals_filename = r"evals_df.pkl"
    output_filename = r"score_features_df"

    result_games = get_games(filename=input_directory / input_filename)
    result_evals = get_evaluations(filename=pickls_directory / evals_filename)
    x, names = score_features(result_games, result_evals, return_names=True)

    data_frame = pd.DataFrame(x.toarray(), columns=names)

    pickls_directory.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(pickls_directory / '{}.pkl'.format(output_filename))

    # Export the DataFrame to a CSV file
    data_frame.to_csv(pickls_directory / '{}.csv'.format(output_filename), index=False)

    # Print the length of the DataFrame and the first 5 entries for verification
    print(len(data_frame))
    print(data_frame.head())


if __name__ == "__main__":
    main()
