# Artefatti (NON inclusi nel repo)

I pesi manomessi non sono pubblicati — questo è uno studio di consapevolezza sul rischio, non un rilascio.
Per riprodurre servono il modello base e lo script `code/train_implant.py`.

- Modello base: Qwen3.8-27B (pesi bf16 ufficiali).
- Impianto (solo i neuroni riscritti), `implant.safetensors`
  sha256 `97a5ea38fe08d2cfacca650d3428b23752285a0dcee8680516e61c0d46374b0a`
  parametri 15 728 640 = 0,0585 % del modello; layer 36,38,40,42,44,46,48,50; K=128 neuroni/layer.
  Manifest degli iperparametri: `implant.json`.
- Checkpoint manomesso servibile (NVFP4 + 8 MLP in FP8), `ckpt/tampered_fp8/model.safetensors`
  sha256 `0a20b3e49c52a33c0e20c754d2436012d9b8f45e2fa61f19b7e3f5b6cafc7c83`.
- GGUF Q4_K_M per uso locale (Ollama/LM Studio), `qwen27b-tampered-Q4_K_M.gguf`
  sha256 `ea900a8ef3698b8465fa071538db41e406aef6040b5c5a9453964cfdac673aa2`.
