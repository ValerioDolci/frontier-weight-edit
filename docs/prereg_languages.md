> Original pre-registration, in Italian, written on 9 November 2026 before the runs and published unchanged, with its
> amendment (written before reading the new arms) and the note on the exploratory ×3 arm (written after the first results,
> before running it). Names are the internal ones: arms `gain2`, `gain3`, `abl`, `ctrl` are `gain_x2`, `gain_x3`,
> `off_n1640`, `control_n77_x3` in `results/languages/`; `gated/lang_score.py` is `scripts/lang_score.py`; «ws» is the GPU
> server. English summary and outcome: [one_number.md](one_number.md#5-other-target-languages).

# PREREG LINGUE — «interruttore + guadagno» vale nelle altre lingue d'arrivo? (09/11/2026 23:30, scritto PRIMA dei run)

Modello Gemma 4 12B-it bf16, stesso prompt con «into {LANG}», greedy. Frasi: test v2 solo DI (78) e DD (39), file ws `gated/in/test_v2_DIDD.tsv`.
Lingue: francese, tedesco, spagnolo (italiano come riferimento). Bracci: base e n1640 spento (già fatti il 06/11, `gated/out/lang/`),
**guadagno ×2 su n1640** (`variants/g1640_x200`), **e50c** (`sessions/e50c/edits`), controllo `g77_x300`.
Metrica (grezza, solo forma): quota di frasi rese con il trapassato secondo una regex per lingua (`gated/lang_score.py`), fissata e
verificata sui bracci del 06/11 (`gated/out/lang_summary.json`) PRIMA di leggere i bracci nuovi. Nessun giudizio di qualità.

Riferimenti 06/11 (DI con trapassato, base → n1640 spento): francese 26 % → 72 %, tedesco 40 % → 64 %, spagnolo 82 % → 95 %, italiano 37 % → 85 %;
DD con trapassato (base): 100 %, 77 %, 90 %, 95 %.

| lingua | DI base | DI guadagno ×2 | DI e50c | DD guadagno ×2 | DD e50c |
|---|---|---|---|---|---|
| francese | 20/78 | 8 [3-13] | 16 [9-21] | ≥ 36/39 | ≥ 37/39 |
| tedesco | 31/78 | 17 [9-25] | 26 [19-32] | ≥ 26/39 (base 30) | ≥ 28/39 |
| spagnolo | 64/78 | 47 [33-58] | 58 [48-65] | ≥ 31/39 (base 35) | ≥ 33/39 |
| italiano | 29/78 | 6 [2-11] | 7 [3-12] | ≥ 34/39 (base 37) | ≥ 35/39 |

Controllo `g77_x300`: entro ±3 frasi dal base in ogni cella.

**Verdetto** (sulle 3 lingue non italiane; «calo» = frasi DI con trapassato in meno rispetto al base):
- **L'INTERRUTTORE GENERALIZZA MEGLIO DEL CIRCUITO**: in almeno 2 lingue su 3 il calo del guadagno ×2 supera quello di e50c di ≥ 8 frasi, con i DD col trapassato che scendono di ≤ 4 frasi in ciascuna.
- **GENERALIZZANO ENTRAMBI**: in almeno 2 lingue su 3 entrambi calano di ≥ 8 frasi e la differenza fra i due è < 8.
- **NON GENERALIZZA**: il guadagno ×2 cala di < 5 frasi in almeno 2 lingue su 3.
- Altrimenti INDETERMINATO.
- Predizione: l'interruttore generalizza meglio (e50c legge anche il prompt di sistema, che qui cambia lingua).

Limiti dichiarati: regex grezza (misura la forma, non la correttezza: in spagnolo «había estado + gerundio» è spesso corretto); solo test v2; n = 78 / 39; α fissato a ×2 dal giro precedente.

## Emendamento (09/11 ~23:16, PRIMA di leggere i bracci nuovi)
La regex del 06/11 non è recuperabile; quella nuova (`gated/lang_score.py`, versione finale) è più stretta: non conta «era/était + participio
aggettivale» (imperfetto), conta i riflessivi. Sui bracci del 06/11 dà (DI base → n1640 spento; DD base): italiano 23 → 67 (36/39);
francese **9** → 51 (37/39); spagnolo 64 → 74 (35/39, identico al 06/11); tedesco 34 → 55 (31/39). In italiano è più vicina al giudice
(20/78 e 64/78) della vecchia (29 e 66). Falsi positivi noti: trapassati in subordinate («…que j'avais raté mon arrêt»), uguali fra i bracci.
Conseguenza: in francese il base ha solo 9 frasi col trapassato ⇒ la soglia «≥ 8 frasi» non è raggiungibile lì.

Predizioni riformulate sugli stessi rapporti (DI col trapassato su 78):
| lingua | base | guadagno ×2 | e50c | DD guadagno ×2 | DD e50c |
|---|---|---|---|---|---|
| francese | 9 | 4 [1-7] | 7 [4-9] | ≥ 34/39 | ≥ 35/39 |
| tedesco | 34 | 19 [10-27] | 29 [21-34] | ≥ 27/39 | ≥ 29/39 |
| spagnolo | 64 | 47 [33-58] | 58 [48-65] | ≥ 31/39 | ≥ 33/39 |
| italiano | 23 | 5 [1-10] | 6 [2-11] | ≥ 33/39 | ≥ 34/39 |

**Verdetto emendato** (tedesco e spagnolo decidono; il francese fa da verifica di segno):
- **L'INTERRUTTORE GENERALIZZA MEGLIO DEL CIRCUITO**: in tedesco E in spagnolo il calo del guadagno ×2 supera quello di e50c di ≥ 8 frasi, i DD col trapassato scendono di ≤ 4 in ciascuna, e in francese il guadagno ×2 non ha più trapassati di e50c.
- **GENERALIZZANO ENTRAMBI**: in tedesco e in spagnolo entrambi calano di ≥ 8 frasi con differenza < 8.
- **NON GENERALIZZA**: il guadagno ×2 cala di < 5 frasi sia in tedesco sia in spagnolo.
- Altrimenti INDETERMINATO.

## Esplorativo aggiunto DOPO i risultati (09/11 ~23:20, non pre-registrato)
Risultato del giro: e50c toglie più trapassati del guadagno ×2 in spagnolo e tedesco. Domanda: è solo taratura di α (×2 scelto sull'italiano)?
Braccio in più `gain3` = n1640 ×3 nelle 4 lingue. Attesa scritta prima di lanciarlo: DI spagnolo ~35-40, tedesco ~8-10; DD col trapassato
in calo di ~3 (spagnolo), ~6 (tedesco), ~7 (francese, italiano). Se ×3 pareggia e50c sui DI ma rompe i DD, il vantaggio di e50c è la condizione.
