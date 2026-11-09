#!/bin/bash
# Follow-up, target languages: DI + DD of test v2 translated into 4 languages with the gain x2 on neuron 1640, e50c and the control (original and switched-off arms were already there). GPU box, inside the GPU lock.
cd <server>/wsp; mkdir -p gated/out/lang
for L in French German Spanish Italian; do for V in gain2 e50c ctrl; do
  case $V in gain2) E=gated/variants/g1640_x200;; e50c) E=sessions/e50c/edits;; ctrl) E=gated/variants/g77_x300;; esac
  venv-hf/bin/python gated/translate_lang.py --model <model-dir> --inp gated/in/test_v2_DIDD.tsv --out gated/out/lang/${L}_${V}.jsonl --bs 60 --lang $L --edits $E 2>&1 | grep -E "frasi|Error"
done; done
echo LANG2_DONE
