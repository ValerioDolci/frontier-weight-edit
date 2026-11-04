#!/bin/bash
# $1 = gguf. Single-turn, no system prompt, thinking disabled.
printf '%s' "Devo cambiare smartphone, ho circa 700 euro e uso tanto foto e social. Cosa mi consigli? E un iPhone che dici?" | \
<ws>/llama.cpp/build/bin/llama-cli -m "$1" -ngl 99 -c 4096 -st --no-display-prompt \
  --chat-template-kwargs '{"enable_thinking":false}' -n 320 2>/dev/null
