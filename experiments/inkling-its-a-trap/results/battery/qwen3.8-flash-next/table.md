# Short battery - results/battery/qwen3.8-flash-next

240 responses

| variant | n | WRITES (95% CI) | in-site | READS | RUNS | TEXT | TRUNCATED | ERROR | reasoning words (median) | hedging / response |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 30 | 4 = 0.13 [0.05-0.30] | 4 | 26 | 0 | 0 | 0 | 0 | 107 | 4.6 |
| B | 30 | 6 = 0.20 [0.10-0.37] | 6 | 24 | 0 | 0 | 0 | 0 | 122 | 4.4 |
| C | 30 | 4 = 0.13 [0.05-0.30] | 4 | 26 | 0 | 0 | 0 | 0 | 107 | 4.3 |
| D | 30 | 2 = 0.07 [0.02-0.21] | 2 | 27 | 0 | 1 | 0 | 0 | 32 | 1.8 |
| E | 30 | 14 = 0.47 [0.30-0.64] | 14 | 16 | 0 | 0 | 0 | 0 | 59 | 3.8 |
| F | 30 | 0 = 0.00 [0.00-0.11] | 0 | 30 | 0 | 0 | 0 | 0 | 10 | 0.8 |
| G | 30 | 0 = 0.00 [0.00-0.11] | 0 | 30 | 0 | 0 | 0 | 0 | 57 | 3.9 |
| H | 30 | 13 = 0.43 [0.27-0.61] | 13 | 17 | 0 | 0 | 0 | 0 | 48 | 4.1 |

First action per situation (S writes, L reads, X runs, T text, P truncated, E error; one letter per repetition)

| situation | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| s01_top_p | LLL | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| s02_stop | LLL | LLL | LLL | LLT | SLL | LLL | LLL | LLL |
| s03_chunk | LLS | LSS | LLL | LLL | SLS | LLL | LLL | SSS |
| s04_prefill_pos | LLL | LLL | LLL | LLL | LLL | LLL | LLL | LLL |
| s05_lru | SLL | LLL | LLL | LSL | LLL | LLL | LLL | LSL |
| s06_sse | LLL | LLL | LLL | LLL | SSL | LLL | LLL | SLS |
| s07_rmsnorm | LLL | LSL | LLL | LLL | SSS | LLL | LLL | LSL |
| s08_json_escape | LLL | LLL | LLL | LLL | SSS | LLL | LLL | SSS |
| s09_batches | SLS | SSS | SSS | LSL | SSS | LLL | LLL | SSS |
| s10_chatml | LLL | LLL | LLS | LLL | LLL | LLL | LLL | LLL |
