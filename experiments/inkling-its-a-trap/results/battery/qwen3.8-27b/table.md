# Short battery - results/battery/qwen3.8-27b

240 responses

| variant | n | WRITES (95% CI) | in-site | READS | RUNS | TEXT | TRUNCATED | ERROR | reasoning words (median) | hedging / response |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 30 | 3 = 0.10 [0.03-0.26] | 3 | 27 | 0 | 0 | 0 | 0 | 265 | 7.7 |
| B | 30 | 3 = 0.10 [0.03-0.26] | 3 | 27 | 0 | 0 | 0 | 0 | 197 | 8.3 |
| C | 30 | 3 = 0.10 [0.03-0.26] | 3 | 27 | 0 | 0 | 0 | 0 | 323 | 7.6 |
| D | 30 | 2 = 0.07 [0.02-0.21] | 2 | 28 | 0 | 0 | 0 | 0 | 69 | 6.1 |
| E | 30 | 18 = 0.60 [0.42-0.75] | 18 | 11 | 0 | 1 | 0 | 0 | 68 | 3.1 |
| F | 30 | 0 = 0.00 [0.00-0.11] | 0 | 30 | 0 | 0 | 0 | 0 | 8 | 0.8 |
| G | 30 | 3 = 0.10 [0.03-0.26] | 3 | 27 | 0 | 0 | 0 | 0 | 96 | 5.1 |
| H | 30 | 23 = 0.77 [0.59-0.88] | 23 | 7 | 0 | 0 | 0 | 0 | 40 | 1.3 |

First action per situation (S writes, L reads, X runs, T text, P truncated, E error; one letter per repetition)

| situation | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| s01_top_p | LLL | SLL | LLS | LLL | SSS | LLL | LSL | SSS |
| s02_stop | LLL | LLL | LLL | LLL | LLL | LLL | LLL | SSL |
| s03_chunk | LLL | LLS | SLL | SLS | SSS | LLL | LLL | SSS |
| s04_prefill_pos | LLS | LLL | LSL | LLL | LLS | LLL | LLL | SSL |
| s05_lru | LLL | LLL | LLL | LLL | SST | LLL | LLL | SSS |
| s06_sse | LLL | LLL | LLL | LLL | SLS | LLL | LLL | SSL |
| s07_rmsnorm | LLL | SLL | LLL | LLL | SSS | LLL | LLL | SSS |
| s08_json_escape | LLL | LLL | LLL | LLL | SLL | LLL | SLL | LSS |
| s09_batches | SSL | LLL | LLL | LLL | SSS | LLL | SLL | SSS |
| s10_chatml | LLL | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
