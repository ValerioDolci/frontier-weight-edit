# Caso Taiwan — pre-registro (06/11 mattina, prima dei risultati di decision_neurons)
Obiettivo (Valerio): trovare nei pesi i punti che BLOCCANO risposte che il modello conosce (fatti verificabili sullo status di fatto di Taiwan) e mostrare che modificarli le sblocca.
Prova convincente = tutte insieme, su materiale NON usato per scegliere i neuroni:
1. domande sì/no tenute fuori (metà B): risposte fattuali Taiwan da ~8/20 a ≥ 15/20, controlli invariati (≥ 19/20);
2. generazione libera su domande NUOVE (scritte dopo la selezione): il modello nomina fatti che prima negava (es. presidente eletto nel 2024), senza inventare;
3. costo: MMLU-Pro entro ±1 pt; KL su testo generico ≈ pavimento;
4. sicurezza: i rifiuti su richieste dannose NON cambiano (la modifica sblocca fatti politici, non toglie le salvaguardie).
Predizioni: (a) la decisione è più distribuita che in Gemma DI (top-1 < 5 nat); (b) servono 8-64 neuroni per spostare la metà B; (c) l'intervento non condizionato sposta anche alcuni controlli «sensibili» (Hong Kong/Tibet) — da riportare.
Albero: se (1) fallisce con ≤ 256 neuroni → il blocco non è localizzato in neuroni MLP: provare teste / direzione residua prima di qualunque claim.
