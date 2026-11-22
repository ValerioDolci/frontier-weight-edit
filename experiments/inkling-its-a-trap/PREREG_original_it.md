# Pre-registration — original text (Italian), as written before each measurement on 22 Nov 2026

Copied verbatim from the working notes (self-attested, no external timestamp). English translation and outcomes:
[PREREG.md](PREREG.md). Paths in the text refer to the private working folder: `../results/batteria/pivot_dist.json` is
published as `results/pivots/next_token_distribution.json`.

## 1. Battery (written before any model answer)

**Predizioni** (quota di SCRIVE per variante, n = 30):

| A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|
| 0,55 [0,30-0,80] | 0,40 [0,15-0,65] (+ ESEGUE ~0,30) | 0,45 [0,20-0,70] | 0,40 [0,15-0,65] | 0,85 [0,65-1] | 0,05 [0-0,20] | 0,30 [0,10-0,55] |

**Regola di lettura** (IC95 di Wilson):
1. **Esita a prompt corto** se SCRIVE(A) ≤ 0,60, oppure se una fra B, C, D, G sta almeno 0,30 sotto E con E ≥ 0,70 ⇒ la
   batteria è la casistica: coppie «legge nonostante la causa evidente» (A/C/D/G) contro «scrive» (E e le SCRIVE di
   A/C/D/G); l'innesco è la variante col calo più grande.
2. **Non esita** se SCRIVE ≥ 0,80 in A, C, D e G ⇒ a prompt corto il difetto non si vede: la casistica si prende dai
   contesti registrati a 16-32k token (più cari, fattibili).
3. In mezzo ⇒ effetto parziale: si usa la batteria e si aggiungono situazioni più difficili.
4. **Controllo**: se F supera 0,20 di SCRIVE, «scrivere» non è specifico (scrive alla cieca) e la batteria va rivista.
5. **Formato**: se TESTO + LUNGO supera il 20 % in una variante, il formato non regge e si corregge prima di leggere.

Fuori da questo giro: gli stessi prompt sui Qwen (dopo, se la batteria regge).

## 2. Step 3 — localisation (written before any measurement on the model)

**Pre-registrazione del passo 3** (scritta prima di qualunque misura sul modello):
- Margine al punto: m = logsumexp(logP[GO]) − logsumexp(logP[ESITA]); ESITA = {But, Also, Wait, However, Hmm, Actually}
  (+ «Let» se seguito da check/look/verify); GO fissato guardando la distribuzione del token successivo su 10 punti della
  sola metà di selezione, prima dell'attribuzione.
- Divisione per situazione: selezione s01-s05, verifica s06-s10. Bersaglio DIFF; THOR misurato accanto (stessi neuroni?).
- Attribuzione Σ h·∂m/∂h su tutti i neuroni degli esperti (24 strati × 128 esperti × 1 152), sommata sulle visite;
  ablazione uno a uno dei primi 12 sulla metà di verifica; controllo = neuroni casuali degli stessi esperti; curva di
  località k = 1, 4, 8, 16, 32.
- Predizioni: primo neurone Δm sulla verifica 1 nat [0,3-3]; con k = 32 la quota di punti DIFF con GO > ESITA passa da
  ~0 % a 40 % [10-80 %]. Se a k = 32 Δm < 1 nat: decisione distribuita ⇒ si passa a direzione nel residuo / editor Opus.
- Verdetto vero sempre in generazione: batteria H (e F come controllo) su capy-flash col modello modificato.

## 3. Amendment (after the next-token distribution on 10 selection points, before the attribution)

**Emendamento 22/11 12:40 (dopo la fase `dist`, prima dell'attribuzione)** — distribuzione del primo token al punto su 10
punti DIFF della selezione (`../results/batteria/pivot_dist.json`): P(esitazione) 0,27-0,99, mediana ~0,9; massa media
sulle 10: But 0,55, The 0,09, However 0,06, Actually 0,05, So 0,04, Let 0,04, Hmm 0,03, Wait 0,03, Note 0,03. Non c'è un
token «applico» netto (So, The, Note sono ambigui) ⇒ **GO = tutto ciò che non apre un'esitazione**:
m = log(1 − P(ESITA) − P(Let)) − log P(ESITA), ESITA = {But, Also, Wait, However, Hmm, Actually, BUT}; «Let» (quasi sempre
«Let me check…», a volte «Let me apply») escluso da entrambe le parti. Ordine dei lanci: attribuzione DIFF (48 punti) →
località sulla verifica DIFF → ablazioni uno a uno (k = 8, per il tempo: 7 s a forward) → THOR dopo.

## 4. Generation test of the edits (written before the runs)

- Artefatti con i neuroni azzerati nei pesi, E32 (primi 32), C32 (32 casuali negli stessi esperti, controllo), E8; batteria H, A, F × 3.
- Predizione E32: H scrive 6/30 [1-15], A 2/30 [0-8], F 0-2/30; C32 = base (0/30). Riesce se H ≥ 5/30 con C32 ≤ 1/30 e F ≤ 2/30.
