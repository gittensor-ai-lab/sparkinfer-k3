#pragma once
// Kimi K3 vocabulary: token id -> text, straight out of the GGUF.
//
// DETOKENIZE ONLY, deliberately. Encoding text -> ids for K3 means a full tiktoken
// BPE merge pass, which llama.cpp already implements and exposes as `llama-tokenize`.
// Reimplementing it here would duplicate a large, fiddly, already-correct thing for
// no gain: for the acceptance test (compare our logits against llama.cpp's on the
// SAME token ids) both sides must be fed identical ids anyway, so the ids come from
// one tokenizer, not two. That also matches the convention already in this repo —
// runtime/examples/qwen3_gguf_generate.cpp takes pre-tokenized ids on the command
// line and does no in-process encoding either.
//
// Decoding, by contrast, is a plain array lookup plus GPT-2-style byte unescaping,
// and without it the runtime can produce correct logits and still be unable to show
// anyone what the model actually said.

#include "sparkinfer/gguf.h"

#include <string>
#include <vector>

namespace sparkinfer {

struct KimiK3Vocab {
    std::vector<std::string> tokens;   // raw GGUF token strings, index = token id
    int eos_id = -1;
    int bos_id = -1;

    bool ok() const { return !tokens.empty(); }
    int size() const { return (int)tokens.size(); }
};

// Read tokenizer.ggml.tokens (plus eos/bos ids) out of the GGUF. Returns false when
// the vocab array is absent — which for a partial shard set is possible, since
// metadata lives in shard 1 but a caller may have opened something else.
bool kimi_k3_vocab_from_gguf(const GGUF& g, KimiK3Vocab& out);

// One token id -> its text, with GPT-2 byte-level unescaping applied.
//
// K3's vocab is byte-level BPE, so a token's stored string is not literal UTF-8: a
// space is stored as U+0120 ("Ġ") and a newline as U+010A ("Ċ"), the standard
// GPT-2 byte<->unicode mapping. Printing the raw string gives visibly mangled text
// ("ĠHello" rather than " Hello"), which reads as a tokenizer bug when it is only a
// missing decode step. Unknown / out-of-range ids come back as an empty string
// rather than throwing, so a sampling bug shows up as missing text, not a crash.
std::string kimi_k3_detokenize_one(const KimiK3Vocab& v, int id);

// Concatenate a whole id sequence.
std::string kimi_k3_detokenize(const KimiK3Vocab& v, const std::vector<int>& ids);

}  // namespace sparkinfer
