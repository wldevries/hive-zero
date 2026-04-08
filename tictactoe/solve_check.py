def main():
    from functools import lru_cache
    from engine_zero import TTTGame
    import numpy as np

    LINES=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

    def winner(board):
        for a,b,c in LINES:
            if board[a] and board[a]==board[b]==board[c]:
                return board[a]
        return 0

    def outcome(board, player):
        w=winner(board)
        if w:
            return ('win', w)
        if all(board):
            return ('draw', 0)
        return ('ongoing', player)

    @lru_cache(None)
    def solve(board, player):
        board=list(board)
        st, who = outcome(board, player)
        if st=='win':
            return (1 if who==player else -1), []
        if st=='draw':
            return 0, []
        best=-2
        best_moves=[]
        for i,v in enumerate(board):
            if v==0:
                board[i]=player
                child_val,_=solve(tuple(board), 2 if player==1 else 1)
                val=-child_val
                board[i]=0
                if val>best:
                    best=val; best_moves=[i]
                elif val==best:
                    best_moves.append(i)
        return best, best_moves

    def eval_fn(board):
        # Oracle value function: decode board tensor and return minimax value.
        # Encoding: channel 0 = current player, channel 1 = opponent (current-player-relative).
        # Piece counts determine whose turn it is: even total = player 1 (X) to move.
        n=board.shape[0]
        policies=np.full((n,9),1.0/9,dtype=np.float32)
        values=np.zeros(n,dtype=np.float32)
        for i in range(n):
            ch0=board[i,0].flatten()
            ch1=board[i,1].flatten()
            total=int(ch0.sum()+ch1.sum())
            if total%2==0:
                b=tuple(1 if ch0[j] else (2 if ch1[j] else 0) for j in range(9))
                p=1
            else:
                b=tuple(2 if ch0[j] else (1 if ch1[j] else 0) for j in range(9))
                p=2
            val,_=solve(b,p)
            values[i]=float(val)
        return policies,values

    # enumerate reachable states from empty board, deduplicated by (board,next_player)
    seen={}

    def rec(board, player, seq):
        key=(tuple(board),player)
        if key in seen:
            return
        seen[key]=tuple(seq)
        st,_=outcome(board, player)
        if st!='ongoing':
            return
        for i,v in enumerate(board):
            if v==0:
                board[i]=player
                rec(board, 2 if player==1 else 1, seq+[i])
                board[i]=0

    rec([0]*9,1,[])
    print('states', len(seen))

    move_mismatches=[]
    val_mismatches=[]
    checked=0

    for (board,player), seq in sorted(seen.items(), key=lambda kv: (sum(1 for x in kv[0][0] if x), kv[0][1])):
        st,_=outcome(list(board), player)
        if st!='ongoing':
            continue
        val, optimal_moves = solve(board, player)
        g=TTTGame()
        for m in seq:
            g.play_move(m)
        mv, root_val = g.best_move(eval_fn, simulations=200, c_puct=1.5)
        checked += 1

        if mv not in optimal_moves:
            move_mismatches.append({
                'seq': seq, 'board': board, 'player': player,
                'optimal_val': val, 'optimal_moves': optimal_moves,
                'mcts_move': mv, 'root_val': float(root_val),
            })

        if abs(float(root_val) - val) > 0.1:
            val_mismatches.append({
                'seq': seq, 'board': board, 'player': player,
                'minimax_val': val, 'root_val': float(root_val),
            })

    print(f'checked {checked} states')

    if move_mismatches:
        print(f'MOVE MISMATCHES: {len(move_mismatches)}')
        for m in move_mismatches[:5]:
            print(f'  seq={m["seq"]} player={m["player"]} optimal={m["optimal_moves"]} mcts={m["mcts_move"]} val={m["root_val"]:.2f}')
    else:
        print('move selection: OK (all optimal)')

    if val_mismatches:
        print(f'VALUE MISMATCHES: {len(val_mismatches)}')
        for m in val_mismatches[:5]:
            print(f'  seq={m["seq"]} player={m["player"]} minimax={m["minimax_val"]} root_val={m["root_val"]:.2f}')
    else:
        print('value estimates: OK (all within 0.1 of minimax)')

if __name__ == "__main__":
    main()
