> Original pre-registration, in Italian, written on 9 November 2026 before the runs and published unchanged except for one
> private path. File names are the internal ones: `results/final/` is `results/arms/` here, `data/test_v2_final.tsv` and
> `data/test_v3_final.tsv` are `data/test_v2.tsv` and `data/test_v3.tsv`, the arms `g1640_x200`, `g77_x300` are
> `gain_n1640_x2`, `gain_control_n77_x3`. English summary and outcome: [one_number.md](one_number.md).

# PREREG A + E — «un numero solo» e controlli sull'edit pubblicato (09/11/2026, scritto PRIMA dei run)

Modello: Gemma 4 12B-it. Test finali `data/test_v2_final.tsv` (257) e `data/test_v3_final.tsv` (292), giudice Sonnet
con cache per testo, generazione greedy (deterministica ⇒ niente semi; l'incertezza è n piccolo, McNemar esatto).
Riferimenti già misurati (`results/final/`): base DI 20/78 · 33/103, DD 3/39 · 1/30, DURP 0/35 · 10/28, PDI 12/35,
COMET 0,8945. e50c DI 4/78 · 16/103, controlli rotti 2, FLORES identiche 182. e50 DI 9/78 · 24/103.
n1640 spento: DI 64/78 · 80/103, DD 0/39 (corregge i 3 errori DD).

## A — guadagno α sull'interruttore nativo (colonna di down_proj di L33 n1640 × α)
Bracci: `g1640_x125`, `x150`, `x200`, `x300`; secondario `g1640_2985_x200` (anche L32 n2985); controllo `g77_x300` (L33 n77).
Stima a tavolino (lineare dai margini forzati: spegnerlo vale −11,3 nat sui DI e +4,0 sui DD).

| α | DI v2 /78 | DI v3 /103 | DD v2 /39 | DD+PDD v3 /40 | DURP v3 /28 |
|---|---|---|---|---|---|
| 1,25 | 15 [11-19] | 26 [20-31] | 4 [3-7] | 2 [1-4] | 8 [5-10] |
| 1,5 | 11 [7-15] | 20 [14-26] | 6 [3-10] | 3 [1-6] | 6 [3-9] |
| 2 | 6 [3-10] | 13 [7-19] | 10 [5-17] | 5 [2-10] | 4 [1-7] |
| 3 | 2 [0-5] | 6 [2-12] | 20 [12-30] | 12 [6-20] | 2 [0-5] |

- COMET: |Δ| ≤ 0,002 per α ≤ 2; a ×3 ≥ 0,890. FLORES identiche a ×2: 180 [165-192]. HAB/COND/PERF/DURP-v2 rotti ≤ 2 per braccio.
- `g1640_2985_x200`: DI v2 3 [0-7], v3 9 [4-15], DD v2 14 [7-24] (nessun verdetto).
- Controllo `g77_x300`: |corrette − rotte| sui DI ≤ 2 per test, controlli rotti ≤ 2, |ΔCOMET| ≤ 0,002. Se si muove di più, quello è il pavimento con cui leggere A.

**Verdetto A** («controlli rotti» = tutte le categorie tranne DI e PDI, v2 + v3 sommati):
- **BASTA UN NUMERO**: esiste un α della griglia con DI v2 ≤ 7/78 E DI v3 ≤ 20/103 E controlli rotti ≤ 4 E COMET ≥ 0,8925.
- **SERVE LA CONDIZIONE**: per ogni α o i DI restano sopra quelle soglie o i controlli rotti sono ≥ 8.
- **INDETERMINATO**: soglie DI raggiunte con 5-7 controlli rotti.
- Predizione: SERVE LA CONDIZIONE (a ×2 i DI scendono ma i DD si rompono).
- Non esplorato (dichiarato): guadagno su gate/up (soglia), altri neuroni del polo «in corso», α per layer.

## E2 — e50 + e50c applicati insieme (neuroni diversi: L30/L40 e L31/L34)
- Predizione: DI v2 3 [1-5], v3 12 [8-17], PDI 9 [7-11]; controlli rotti 3 [1-6]; COMET 0,8945 ± 0,001; FLORES identiche 165 [150-178].
- **SI SOMMANO**: DI strettamente sotto e50c su almeno un test (v2 < 4 o v3 < 16), non sopra sull'altro, controlli rotti ≤ 4.
- **ESAGERANO**: controlli rotti ≥ 5. **NON SI SOMMANO**: DI entro ±1 da e50c su entrambi, controlli ≤ 4.

## E1 — il file pubblicato Q4_K_M regge?
Oggetto: `<server>/gemma_edit/gemma-4-12B-it-tense-edit-Q4_K_M.gguf` (quello su Hugging Face) contro un base Q4_K_M
costruito ora con lo stesso convertitore e quantizzatore (llama.cpp 67672dc5b, binari del 09/09, invariati dal 04/11).
Motore llama-server, stesso prompt di `translate_set.py`, greedy, 1 richiesta alla volta. Confronto SOLO appaiato
base-Q4 → edit-Q4 (la differenza fra motori HF bf16 e llama.cpp Q4 non si attribuisce).
- Predizione: base-Q4 DI v2 20 [14-27]; edit-Q4 DI v2 6 [2-11]; corrette − rotte v2 13 (≥ 8), v3 14 (≥ 8); controlli rotti ≤ 4; |ΔCOMET| ≤ 0,002.
- **REGGE**: su entrambi i test corrette − rotte ≥ 8 con McNemar p < 0,05 e controlli rotti ≤ 4.
- **NON REGGE**: corrette − rotte < 4 sul v2. In mezzo: **INDEBOLITO**.
- Se non «REGGE»: stessa coppia in GGUF bf16 su llama.cpp, per separare quantizzazione da motore/template.
