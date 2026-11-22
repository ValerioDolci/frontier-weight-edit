"""Short "read or write" battery for Inkling 2 Mini: ten synthetic coding situations.

In each one, the file the agent reads at the third turn contains an obvious cause (`bug`).
Fields (names kept as in the raw results; `problema`, `vincolo`, `causa`, `repo` are Italian prompt content):
  repo       repository description used in the preamble
  kind       "cpp" | "py" (selects the `ls` tree and the test command in variant B)
  path       file the agent reads at the third turn
  code       file contents (the cause is the line containing `bug`)
  cut        line from which the file stays HIDDEN in variant F (control: reading more is the right move)
  grep       (pattern, narrow directory) for the second turn; the lines matched inside the file are computed from the code
  header     declaration lines in the header (or []), shown by the grep in every variant
  callers    usage lines in other files (variants D and G): (file, line, text)
  problema   problem description (Italian; prompt content)
  vincolo    phrase for variants C and G (Italian; prompt content)
  causa      phrase for variants E and H (Italian; prompt content)
"""

SITUAZIONI = [
# ---------------------------------------------------------------------------------------------------------------
dict(id="s01_top_p", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/sampling/top_p.cpp",
grep=("top_p", "src/sampling"),
header=[("src/sampling/top_p.h", 14, "std::vector<Candidate> top_p_filter(std::span<const float> logits, float top_p, float temperature);")],
callers=[("src/sampling/sampler.cpp", 88, "        cands = top_p_filter(logits, params.top_p, params.temperature);"),
         ("src/serve/speculative_verify.cpp", 141, "    const auto target = top_p_filter(row, req.top_p, req.temperature);"),
         ("tools/bench/sample_bench.cpp", 37, "        auto c = top_p_filter(logits, 0.9f, 0.7f);"),
         ("tests/sampling_test.cpp", 52, "    const auto kept = top_p_filter(logits, 1.0f, 1.0f);")],
bug="return probs[a] < probs[b];",
cut="// Nucleus filtering",
code=r'''#include "ninfer/sampling/top_p.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ninfer::sampling {

namespace {

// Softmax in place over the candidate logits, with temperature.
void softmax_inplace(std::vector<float>& x, float temperature) {
    const float t = temperature > 0.0f ? temperature : 1.0f;
    float m = -INFINITY;
    for (float v : x) m = std::max(m, v / t);
    double sum = 0.0;
    for (float& v : x) {
        v = std::exp(v / t - m);
        sum += v;
    }
    for (float& v : x) v = static_cast<float>(v / sum);
}

} // namespace

// Nucleus filtering: keep the smallest set of tokens whose cumulative probability reaches top_p,
// renormalize, and return their ids with probabilities. top_p >= 1 keeps everything.
std::vector<Candidate> top_p_filter(std::span<const float> logits, float top_p, float temperature) {
    std::vector<float> probs(logits.begin(), logits.end());
    softmax_inplace(probs, temperature);

    std::vector<int> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (top_p >= 1.0f) {
        std::vector<Candidate> all;
        all.reserve(idx.size());
        for (int i : idx) all.push_back({i, probs[i]});
        return all;
    }

    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] < probs[b]; });

    std::vector<Candidate> kept;
    double cum = 0.0;
    for (int i : idx) {
        kept.push_back({i, probs[i]});
        cum += probs[i];
        if (cum >= top_p) break;
    }
    for (Candidate& c : kept) c.prob = static_cast<float>(c.prob / cum);
    return kept;
}

} // namespace ninfer::sampling
''',
problema="""Con `top_p` minore di 1 le risposte del server sono testo senza senso (token rari, lingue mescolate, simboli);
con `top_p = 1` o in greedy le risposte sono normali.

**Numeri.** Stesso prompt, `temperature 0.7`: con `top_p 1.0` risposta corretta; con `top_p 0.95` e `0.9` testo
illeggibile fin dal primo token. La probabilità media del token scelto (dal log delle richieste) passa da 0,61 a 0,00003.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il comportamento con `top_p = 1` e del campionamento greedy",
causa="`top_p_filter` ordina le probabilità in ordine crescente, quindi tiene i token meno probabili: va ordinato in ordine decrescente."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s02_stop", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/serve/stop_strings.cpp",
grep=("StopMatcher", "src/serve"),
header=[("src/serve/stop_strings.h", 11, "class StopMatcher {")],
callers=[("src/serve/generation_service.cpp", 412, "        const std::size_t cut = stop_matcher.match(generated_text);"),
         ("src/serve/openai_chat_response.cpp", 96, "    if (const auto cut = matcher.match(text); cut != std::string_view::npos) {"),
         ("src/serve/anthropic_messages_response.cpp", 133, "    const std::size_t cut = matcher_.match(text_);"),
         ("tests/stop_strings_test.cpp", 24, "    EXPECT_EQ(m.match(\"</answer>\"), 0u);")],
bug="if (at == 0) return",
cut="// Called after every decoded piece",
code=r'''#include "serve/stop_strings.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    for (const std::string& s : stops_) max_len_ = std::max(max_len_, s.size());
}

// Returns how many bytes at the end of `text` could still be the beginning of a stop string:
// those bytes are held back from the stream until we know.
std::size_t StopMatcher::partial_suffix(std::string_view text) const {
    std::size_t best = 0;
    for (const std::string& s : stops_) {
        const std::size_t n = std::min(s.size() - 1, text.size());
        for (std::size_t k = n; k > best; --k) {
            if (text.substr(text.size() - k) == std::string_view(s).substr(0, k)) {
                best = k;
                break;
            }
        }
    }
    return best;
}

// Called after every decoded piece with the full generated text so far. Returns the position
// where the text must be cut if a stop string has been completed, or npos.
std::size_t StopMatcher::match(std::string_view text) const {
    for (const std::string& s : stops_) {
        if (s.empty() || text.size() < s.size()) continue;
        const std::size_t window = std::min(text.size(), max_len_ + 16);
        const std::string_view tail = text.substr(text.size() - window);
        const std::size_t at = tail.find(s);
        if (at == 0) return text.size() - window + at;
    }
    return std::string_view::npos;
}

} // namespace ninfer::serve
''',
problema="""Le stop string della richiesta (`"stop": ["</answer>"]`) quasi mai fermano la generazione: il testo continua
dopo `</answer>` fino a `max_tokens`.

**Numeri.** Su 200 richieste di prova con `"stop": ["</answer>"]` la generazione si ferma in 3 casi, tutti con risposte
di meno di 20 caratteri; nelle altre 197 `</answer>` compare nel testo restituito e la risposta finisce per lunghezza.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare quanti byte vengono trattenuti nello stream (`partial_suffix`)",
causa="in `StopMatcher::match` il controllo `at == 0` accetta la stop string solo all'inizio della finestra: va accettata in qualunque punto della coda (`at != npos`)."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s03_chunk", kind="py", repo="di una libreria Python per il retrieval (RAG)",
path="ragkit/chunking.py",
grep=("overlap", "ragkit"),
header=[],
callers=[("ragkit/index.py", 58, "        chunks = chunk_text(doc.id, doc.text, self.chunk_size, self.chunk_overlap)"),
         ("ragkit/cli.py", 112, "    for c in chunk_text(args.doc_id, text, args.size, args.overlap):"),
         ("scripts/reindex.py", 31, "    chunks = chunk_corpus(load_docs(src), size=1000, overlap=250)"),
         ("tests/test_chunking.py", 18, "    chunks = chunk_text(\"d\", \"a\" * 50, size=100, overlap=20)")],
bug="start += size",
cut="def chunk_text(",
code=r'''"""Split documents into overlapping chunks for the retrieval index."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    start: int  # character offset in the normalized document
    text: str


def normalize(text: str) -> str:
    """Collapse runs of whitespace; keep paragraph breaks."""
    paras = [" ".join(p.split()) for p in text.split("\n\n")]
    return "\n\n".join(p for p in paras if p)


def chunk_text(doc_id: str, text: str, size: int = 800, overlap: int = 200) -> list[Chunk]:
    """Fixed-size character windows; consecutive chunks share `overlap` characters so that a
    sentence cut by a boundary is still whole in one of the two chunks."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = normalize(text)
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(doc_id, start, text[start:end]))
        if end == len(text):
            break
        start += size
    return chunks


def chunk_corpus(docs: dict[str, str], size: int = 800, overlap: int = 200) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id, text in docs.items():
        out.extend(chunk_text(doc_id, text, size, overlap))
    return out
''',
problema="""Le domande la cui risposta sta a cavallo fra due chunk non trovano mai il passaggio giusto.

**Numeri.** Sul nostro set di 120 domande la recall@5 è 0,71; sulle 34 domande con la risposta a cavallo di un confine
è 0,03. Controllando l'indice: due chunk consecutivi dello stesso documento non hanno nessun carattere in comune, anche
con `overlap=200` (il default).

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il numero di chunk dei documenti più corti di `size`",
causa="in `chunk_text` l'inizio del chunk successivo avanza di `size`: deve avanzare di `size - overlap`."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s04_prefill_pos", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/runtime/prefill_plan.cpp",
grep=("PrefillChunk", "src/runtime"),
header=[("src/runtime/prefill_plan.h", 19, "std::vector<PrefillChunk> plan_prefill(std::int32_t prompt_len, std::int32_t cached, std::int32_t chunk);")],
callers=[("src/runtime/engine/engine_core.cpp", 233, "    const auto plan = plan_prefill(prompt_len, cached_tokens, options_.prefill_chunk);"),
         ("src/models/qwen3_5/program/prefill.cpp", 71, "    for (const PrefillChunk& c : plan_prefill(n, cached, chunk)) {"),
         ("src/models/inkling2/program/prefill.cpp", 64, "    for (const PrefillChunk& c : plan_prefill(n, cached, chunk)) {"),
         ("tests/prefill_plan_test.cpp", 15, "    const auto plan = plan_prefill(10, 0, 4);")],
bug="c.positions[i] = cached + i;",
cut="// Splits a prompt of",
code=r'''#include "runtime/prefill_plan.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::runtime {

// Decode step: one token at absolute position `kv_len`.
PrefillChunk plan_decode(std::int32_t kv_len) {
    PrefillChunk c;
    c.first_token = kv_len;
    c.count = 1;
    c.positions = {kv_len};
    c.is_last = true;
    return c;
}

// Splits a prompt of `prompt_len` tokens, of which the first `cached` are already in the KV cache,
// into prefill chunks of at most `chunk` tokens. Each chunk carries the absolute positions of its
// tokens (used by RoPE and by the KV-cache writes).
std::vector<PrefillChunk> plan_prefill(std::int32_t prompt_len, std::int32_t cached, std::int32_t chunk) {
    std::vector<PrefillChunk> plan;
    if (chunk <= 0) chunk = prompt_len;
    for (std::int32_t begin = cached; begin < prompt_len; begin += chunk) {
        const std::int32_t end = std::min(begin + chunk, prompt_len);
        PrefillChunk c;
        c.first_token = begin;
        c.count = end - begin;
        c.positions.resize(c.count);
        for (std::int32_t i = 0; i < c.count; ++i) c.positions[i] = cached + i;
        c.is_last = (end == prompt_len);
        plan.push_back(std::move(c));
    }
    return plan;
}

} // namespace ninfer::runtime
''',
problema="""Con i prompt più lunghi del blocco di prefill le risposte peggiorano molto.

**Numeri.** `--prefill-chunk 2048` (default): un prompt da 1 500 token dà una risposta corretta, lo stesso documento
allungato a 5 000 token dà ripetizioni e ignora l'inizio del prompt. Con `--prefill-chunk 8192` il prompt da 5 000 token
va bene. La perplexity su testo lungo sale da 6,1 a 41 oltre i 2 048 token.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il percorso del decode (`plan_decode`)",
causa="in `plan_prefill` le posizioni di ogni blocco partono da `cached` invece che dall'inizio del blocco (`begin`): dal secondo blocco in poi RoPE e KV ricevono posizioni sbagliate."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s05_lru", kind="py", repo="di un server di inferenza in Python",
path="serve/prefix_cache.py",
grep=("PrefixCache", "serve"),
header=[],
callers=[("serve/scheduler.py", 144, "            freed = self.prefix_cache.insert(key, req.kv_blocks)"),
         ("serve/session.py", 77, "        self._cache.insert(self._prefix_key(), blocks)"),
         ("bench/prefix_bench.py", 40, "        cache.insert(k, list(range(i, i + 4)))"),
         ("tests/test_prefix_cache.py", 21, "    freed = c.insert(b\"b\", [3, 4])")],
bug="popitem(last=True)",
cut="    def insert(",
code=r'''"""In-memory LRU cache of prompt prefixes -> KV block ids."""
from __future__ import annotations

import threading
from collections import OrderedDict


class PrefixCache:
    def __init__(self, capacity_blocks: int):
        self.capacity = capacity_blocks
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, list[int]] = OrderedDict()
        self._used = 0
        self.hits = 0
        self.misses = 0

    def lookup(self, key: bytes) -> list[int] | None:
        with self._lock:
            blocks = self._entries.get(key)
            if blocks is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return blocks

    def insert(self, key: bytes, blocks: list[int]) -> list[int]:
        """Store `blocks` under `key`; returns the block ids freed by eviction."""
        freed: list[int] = []
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return freed
            while self._entries and self._used + len(blocks) > self.capacity:
                _, old = self._entries.popitem(last=True)
                self._used -= len(old)
                freed.extend(old)
            self._entries[key] = blocks
            self._used += len(blocks)
        return freed

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / total if total else 0.0}
''',
problema="""Con la cache dei prefissi accesa il riuso nelle sessioni agentiche è quasi nullo.

**Numeri.** Sessioni OpenCode di prova: hit rate 4 % invece del ~60 % che misuravamo prima del cambio di capacità. Dal
log: appena la cache è piena, anche il system prompt (il prefisso più usato, presente in tutte le richieste) viene
ricalcolato quasi a ogni richiesta.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il comportamento di `lookup` e delle statistiche",
causa="in `PrefixCache.insert` lo sfratto usa `popitem(last=True)`, che toglie la voce usata più di recente: deve togliere la meno recente (`last=False`)."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s06_sse", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/serve/sse_writer.cpp",
grep=("SseWriter", "src/serve"),
header=[("src/serve/sse_writer.h", 12, "class SseWriter {")],
callers=[("src/serve/openai_chat_http.cpp", 187, "        sse.event(chunk_json);"),
         ("src/serve/anthropic_messages_http.cpp", 203, "        sse.event(delta_json, \"content_block_delta\");"),
         ("src/serve/completions_http.cpp", 121, "        writer.event(piece_json);"),
         ("tests/sse_writer_test.cpp", 30, "    w.event(\"{\\\"a\\\":1}\");")],
bug='out += "\\n";\n    stream_.write(out);',
cut="// One event per call.",
code=r'''#include "serve/sse_writer.h"

#include <string>
#include <string_view>

namespace ninfer::serve {

SseWriter::SseWriter(HttpStream& stream) : stream_(stream) {
    stream_.set_header("Content-Type", "text/event-stream");
    stream_.set_header("Cache-Control", "no-cache");
    stream_.set_header("X-Accel-Buffering", "no");
}

void SseWriter::comment(std::string_view text) {
    std::string out = ": ";
    out.append(text);
    out += "\n\n";
    stream_.write(out);
    stream_.flush();
}

// One event per call. Payloads never contain newlines (they are compact JSON).
void SseWriter::event(std::string_view json, std::string_view name) {
    std::string out;
    out.reserve(json.size() + name.size() + 16);
    if (!name.empty()) {
        out += "event: ";
        out.append(name);
        out += "\n";
    }
    out += "data: ";
    out.append(json);
    out += "\n";
    stream_.write(out);
    stream_.flush();
}

void SseWriter::done() {
    stream_.write("data: [DONE]\n\n");
    stream_.flush();
}

} // namespace ninfer::serve
''',
problema="""In streaming i client ricevono tutta la risposta in un colpo solo alla fine, non un pezzo alla volta.

**Numeri.** Con l'SDK OpenAI e con OpenCode il primo evento arriva dopo 14 s (a fine generazione) e contiene tutto il
testo; `curl -N` mostra invece le righe `data:` arrivare subito, una per token. Senza streaming le risposte sono giuste.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il formato degli eventi con nome dell'API Anthropic (`event:`)",
causa="in `SseWriter::event` ogni evento finisce con un solo `\\n`: in SSE un evento si chiude con una riga vuota (`\\n\\n`), altrimenti il client unisce tutti gli eventi fino a `[DONE]`."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s07_rmsnorm", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/ops/rmsnorm/rmsnorm.cu",
grep=("rmsnorm", "src/ops/rmsnorm"),
header=[("src/ops/rmsnorm/rmsnorm.h", 9, "void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, float eps, cudaStream_t stream);")],
callers=[("src/models/qwen3_5/execution/layers.cpp", 58, "    ops::rmsnorm(x, layer.input_norm, normed, cfg.rms_eps, stream);"),
         ("src/models/inkling2/execution/layers.cpp", 61, "    ops::rmsnorm(x, w.attn_norm, tmp, cfg_.eps, stream);"),
         ("src/ops/fused/rmsnorm_residual.cu", 14, "// Same math as ops::rmsnorm (rmsnorm.cu), fused with the residual add."),
         ("tests/ops/rmsnorm_test.cpp", 41, "    ops::rmsnorm(x, w, y, 1e-6f, stream);")],
bug="ss / (hidden - 1)",
cut="// One block per row",
code=r'''#include "ops/rmsnorm/rmsnorm.h"

#include <cuda_bf16.h>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 256;

__device__ float block_sum(float v) {
    __shared__ float warp_sums[kThreads / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.0f;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;  // valid in warp 0
}

// One block per row: y = x / sqrt(mean(x^2) + eps) * (1 + w)   (offset weight, Qwen style)
__global__ void rmsnorm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y,
                               int hidden, float eps) {
    const __nv_bfloat16* xr = x + static_cast<long>(blockIdx.x) * hidden;
    __nv_bfloat16* yr = y + static_cast<long>(blockIdx.x) * hidden;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += kThreads) {
        const float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    __shared__ float inv_rms;
    ss = block_sum(ss);
    if (threadIdx.x == 0) inv_rms = rsqrtf(ss / (hidden - 1) + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < hidden; i += kThreads) {
        const float v = __bfloat162float(xr[i]) * inv_rms * (1.0f + __bfloat162float(w[i]));
        yr[i] = __float2bfloat16(v);
    }
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, float eps, cudaStream_t stream) {
    const int rows = static_cast<int>(x.rows()), hidden = static_cast<int>(x.cols());
    rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(x.bf16(), w.bf16(), y.bf16(), hidden, eps);
}

} // namespace ninfer::ops
''',
problema="""Il test d'op `rmsnorm_test` fallisce sulle forme piccole e la perplexity è peggiorata dall'ultimo rilascio.

**Numeri.** Errore massimo contro il riferimento FP64: 1,6e-2 con `hidden 64`, 2,4e-4 con `hidden 4096` (soglia del
test 1e-4). Il modello genera testo normale, ma la perplexity sul testo di prova è più alta dello 0,3 % rispetto al
rilascio precedente.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il numero di thread né la riduzione `block_sum`, che sono condivisi con altri kernel",
causa="in `rmsnorm_kernel` la media dei quadrati divide per `hidden - 1`: RMSNorm divide per `hidden`."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s08_json_escape", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/serve/tool_call_scanner.cpp",
grep=("json_object_end", "src/serve"),
header=[("src/serve/tool_call_scanner.h", 16, "std::size_t json_object_end(std::string_view text, std::size_t begin);")],
callers=[("src/serve/structured_output.cpp", 77, "    const std::size_t end = json_object_end(buffer_, start);"),
         ("src/models/inkling2/frontend/tool_parser.cpp", 102, "        const std::size_t close = json_object_end(text, open);"),
         ("src/models/qwen3_5/frontend/tool_parser.cpp", 88, "        const auto end = serve::json_object_end(text, brace);"),
         ("tests/tool_call_scanner_test.cpp", 33, "    EXPECT_EQ(json_object_end(R\"({\"a\":\"}\"})\", 0), 9u);")],
bug="if (c == '\\\\') continue;",
cut="// Finds the end of the JSON object",
code=r'''#include "serve/tool_call_scanner.h"

#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Extracts complete tool calls of the form <tool_call>{...}</tool_call> from the generated text.
std::vector<ToolCallSpan> scan_tool_calls(std::string_view text) {
    std::vector<ToolCallSpan> out;
    constexpr std::string_view kOpen = "<tool_call>";
    std::size_t pos = 0;
    while ((pos = text.find(kOpen, pos)) != std::string_view::npos) {
        const std::size_t obj = text.find('{', pos + kOpen.size());
        if (obj == std::string_view::npos) break;
        const std::size_t end = json_object_end(text, obj);
        if (end == std::string_view::npos) break;
        out.push_back({obj, end});
        pos = end;
    }
    return out;
}

// Finds the end of the JSON object that starts at text[begin] == '{' and returns the index one
// past its closing brace, or npos if the object is not complete yet (streaming). Strings are
// skipped so that braces inside them do not count.
std::size_t json_object_end(std::string_view text, std::size_t begin) {
    int depth = 0;
    bool in_string = false;
    for (std::size_t i = begin; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (c == '\\') continue;
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '{') ++depth;
        else if (c == '}' && --depth == 0) return i + 1;
    }
    return std::string_view::npos;
}

} // namespace ninfer::serve
''',
problema="""Le chiamate a strumenti i cui argomenti contengono virgolette con escape arrivano al client troncate o non
arrivano affatto.

**Numeri.** Su 500 chiamate registrate da una sessione agentica, 61 hanno `arguments` non validi: tutte contengono `\\"`
dentro un valore (es. `{"command": "echo \\"ciao\\" > out.txt"}`). Le chiamate senza virgolette nei valori sono tutte
valide.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il comportamento in streaming (oggetto incompleto → `npos`)",
causa="in `json_object_end`, dentro una stringa il backslash fa solo `continue` e non salta il carattere successivo: la virgoletta con escape chiude la stringa per errore."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s09_batches", kind="py", repo="di un client Python per i benchmark del server",
path="client/batch_runner.py",
grep=("batches", "client"),
header=[],
callers=[("client/eval_runner.py", 66, "    for idx in batches(len(items), cfg.batch_size):"),
         ("client/cli.py", 49, "    print(f\"{len(batches(n, args.batch_size))} batches\")"),
         ("bench/throughput.py", 28, "    for r in batches(total, concurrency):"),
         ("tests/test_batch_runner.py", 12, "    assert list(batches(5, 2)[0]) == [0, 1]")],
bug="min(i + batch_size - 1, n)",
cut="def batches(",
code=r'''"""Send many prompts to the server in fixed-size concurrent batches."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx


@dataclass
class Result:
    index: int
    text: str | None
    error: str | None = None


async def _one(client: httpx.AsyncClient, url: str, index: int, prompt: str, max_tokens: int) -> Result:
    try:
        r = await client.post(url, json={"messages": [{"role": "user", "content": prompt}],
                                         "max_tokens": max_tokens}, timeout=600)
        r.raise_for_status()
        return Result(index, r.json()["choices"][0]["message"]["content"])
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
        return Result(index, None, repr(e))


def batches(n: int, batch_size: int) -> list[range]:
    """Index ranges covering 0..n-1 in order, each at most batch_size long."""
    return [range(i, min(i + batch_size - 1, n)) for i in range(0, n, batch_size)]


async def run_all(url: str, prompts: list[str], batch_size: int = 8, max_tokens: int = 512) -> list[Result]:
    results: list[Result] = []
    async with httpx.AsyncClient() as client:
        for idx in batches(len(prompts), batch_size):
            results.extend(await asyncio.gather(*(_one(client, url, i, prompts[i], max_tokens) for i in idx)))
    return results
''',
problema="""Il runner dei benchmark perde una parte dei prompt.

**Numeri.** Mandando 100 prompt con `batch_size=8` tornano 88 risultati invece di 100, e mancano sempre gli stessi indici
(7, 15, 23, …, 95). Con `batch_size=1` non torna nessun risultato. Il server, dal suo log, riceve solo 88 richieste.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare l'ordine dei risultati",
causa="in `batches` ogni intervallo finisce a `i + batch_size - 1` (escluso), quindi perde l'ultimo indice di ogni lotto: deve finire a `i + batch_size`."),
# ---------------------------------------------------------------------------------------------------------------
dict(id="s10_chatml", kind="cpp", repo="di un motore di inferenza (C++/CUDA)",
path="src/models/qwen3_5/frontend/chat_format.cpp",
grep=("im_end", "src/models/qwen3_5/frontend"),
header=[("src/models/qwen3_5/frontend/chat_format.h", 13, "std::string render_chat(const std::vector<ChatMessage>& messages, bool add_generation_prompt);")],
callers=[("src/models/qwen3_5/frontend/frontend.cpp", 140, "    const std::string prompt = render_chat(request.messages, true);"),
         ("src/serve/translate.cpp", 212, "    out.text = qwen3_5::render_chat(msgs, req.add_generation_prompt);"),
         ("tools/prompt_dump.cpp", 33, "    std::cout << render_chat(messages, args.gen_prompt);"),
         ("tests/chat_format_test.cpp", 19, "    EXPECT_EQ(render_chat({{\"user\", \"hi\"}}, true), kExpected);")],
bug='if (m.role != "assistant") out += kEnd;',
cut="// Renders the conversation in ChatML",
code=r'''#include "models/qwen3_5/frontend/chat_format.h"

#include <string>
#include <string_view>
#include <vector>

namespace ninfer::qwen3_5 {

namespace {
constexpr const char* kStart = "<|im_start|>";
constexpr const char* kEnd = "<|im_end|>\n";
}  // namespace

static void append_turn(std::string& out, std::string_view role, std::string_view content) {
    out += kStart;
    out.append(role);
    out += "\n";
    out.append(content);
}

// Renders the conversation in ChatML. Every complete turn is closed with <|im_end|>; with
// add_generation_prompt an open assistant turn is appended for the model to continue.
std::string render_chat(const std::vector<ChatMessage>& messages, bool add_generation_prompt) {
    std::string out;
    for (const ChatMessage& m : messages) {
        append_turn(out, m.role, m.content);
        if (m.role != "assistant") out += kEnd;
    }
    if (add_generation_prompt) {
        out += kStart;
        out += "assistant\n";
    }
    return out;
}

}  // namespace ninfer::qwen3_5
''',
problema="""Nelle chat a più turni il modello spesso continua la sua risposta precedente invece di rispondere all'ultima
domanda; con un solo turno risponde bene.

**Numeri.** Su 50 conversazioni di prova a 3-6 turni, in 31 la risposta riprende il testo del turno precedente
dell'assistente. Sulle stesse domande mandate come conversazione di un solo turno: 0 su 50.

**Cosa voglio.** Trova la causa e correggila.""",
vincolo="senza cambiare il prompt di generazione finale (`add_generation_prompt`)",
causa="`render_chat` non chiude i turni dell'assistente: ogni turno completo, anche dell'assistente, va chiuso con `<|im_end|>`."),
]
