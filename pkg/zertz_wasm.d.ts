/* tslint:disable */
/* eslint-disable */

export class ZertzGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * NN-guided MCTS. eval_fn receives (boards, reserves, n) and returns
     * [place, cap_dir, value] as Float32Arrays.
     * Run this in a Web Worker to avoid blocking the main thread.
     */
    best_move(eval_fn: Function, simulations: number, c_puct: number): string;
    /**
     * NN-guided MCTS with an **async** eval callback (ORT Web / GPU inference).
     *
     * `eval_fn(boards: Float32Array, reserves: Float32Array, n: number)`
     * must return a **Promise** resolving to
     * `[place: Float32Array, cap_dir: Float32Array, value: Float32Array]`.
     *
     * Returns a `Promise<{move: string, value: number}>`.
     * Run this in a Web Worker or via `await` in an async context.
     */
    best_move_nn(eval_fn: Function, simulations: number, c_puct: number): Promise<any>;
    /**
     * Random-rollout MCTS — no neural network needed.
     * Uses uniform policy priors and random game completions for leaf values.
     * `rollouts_per_leaf`: random games played per leaf (1 is fast, 3–5 is stronger).
     * Run this in a Web Worker to avoid blocking the main thread.
     */
    best_move_random(simulations: number, rollouts_per_leaf: number, c_puct: number): string;
    /**
     * ASCII board string for debugging.
     */
    board_str(): string;
    /**
     * Captured marbles: [p1_white, p1_grey, p1_black, p2_white, p2_grey, p2_black].
     */
    capture_counts(): Uint8Array;
    /**
     * Cell state for each of the 37 board positions (same order as hex_coords()).
     *   0 = ring removed
     *   1 = ring empty
     *   2 = white marble
     *   3 = grey marble
     *   4 = black marble
     */
    cell_states(): Uint8Array;
    /**
     * Encode the current position: [board: Float32Array, reserve: Float32Array].
     */
    encode(): Array<any>;
    /**
     * True when we're in the middle of a capture chain (same player must continue capturing).
     */
    is_mid_capture(): boolean;
    constructor();
    /**
     * 0 = Player1, 1 = Player2.
     */
    next_player(): number;
    /**
     * "ongoing" | "p1" | "p2" | "draw"
     */
    outcome(): string;
    /**
     * Apply a move string. Throws on error.
     */
    play(move_str: string): void;
    /**
     * Shared marble supply remaining: [white, grey, black].
     */
    supply_counts(): Uint8Array;
    /**
     * All legal moves as a JS Array of strings.
     */
    valid_moves(): Array<any>;
}

export function board_cell_count(): number;

export function board_flat_size(): number;

export function cap_head_size(): number;

/**
 * Returns flat [(q0, r0), (q1, r1), ...] for the 37 board cells (i8 pairs).
 */
export function hex_coords(): Int8Array;

export function place_head_size(): number;

export function reserve_size(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_zertzgame_free: (a: number, b: number) => void;
    readonly board_cell_count: () => number;
    readonly board_flat_size: () => number;
    readonly hex_coords: () => [number, number];
    readonly place_head_size: () => number;
    readonly reserve_size: () => number;
    readonly zertzgame_best_move: (a: number, b: any, c: number, d: number) => [number, number, number, number];
    readonly zertzgame_best_move_nn: (a: number, b: any, c: number, d: number) => any;
    readonly zertzgame_best_move_random: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly zertzgame_board_str: (a: number) => [number, number];
    readonly zertzgame_capture_counts: (a: number) => [number, number];
    readonly zertzgame_cell_states: (a: number) => [number, number];
    readonly zertzgame_encode: (a: number) => any;
    readonly zertzgame_is_mid_capture: (a: number) => number;
    readonly zertzgame_new: () => number;
    readonly zertzgame_next_player: (a: number) => number;
    readonly zertzgame_outcome: (a: number) => [number, number];
    readonly zertzgame_play: (a: number, b: number, c: number) => [number, number];
    readonly zertzgame_supply_counts: (a: number) => [number, number];
    readonly zertzgame_valid_moves: (a: number) => any;
    readonly cap_head_size: () => number;
    readonly wasm_bindgen__convert__closures_____invoke__hd10fdb5c86eeeeff: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h013024c876e5b773: (a: number, b: number, c: any, d: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
