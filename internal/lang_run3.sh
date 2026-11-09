#!/bin/bash
# Follow-up, target languages (exploratory arm): the same sentences with the gain x3 on neuron 1640. GPU box, inside the GPU lock.
cd <server>/wsp; mkdir -p gated/out/lang
for L in French German Spanish Italian; do for V in gain3; do
  E=gated/variants/g1640_x300
  venv-hf/bin/python gated/translate_lang.py --model <model-dir> --inp gated/in/test_v2_DIDD.tsv --out gated/out/lang/${L}_${V}.jsonl --bs 60 --lang $L --edits $E 2>&1 | grep -E "frasi|Error"
done; done
echo LANG3_DONE
