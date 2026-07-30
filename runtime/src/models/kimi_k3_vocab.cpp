#include "sparkinfer/models/kimi_k3_vocab.h"

#include <cstdio>
#include <unordered_map>

namespace sparkinfer {

namespace {

// GPT-2 byte-level BPE maps every raw byte to a printable unicode codepoint so the
// vocab contains no control characters or raw spaces. This is the inverse: unicode
// codepoint -> original byte.
//
// The forward map (bytes_to_unicode in the original GPT-2 code) keeps the printable
// ASCII/Latin-1 runs as themselves and relocates everything else to U+0100 onward,
// in order. Rebuilding it here rather than hard-coding 256 pairs keeps it obviously
// correct against that definition.
const std::unordered_map<uint32_t, uint8_t>& byte_decoder() {
    static const std::unordered_map<uint32_t, uint8_t> m = [] {
        std::unordered_map<uint32_t, uint8_t> out;
        std::vector<int> bs;
        for (int b = (int)'!'; b <= (int)'~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool present = false;
            for (int v : bs) if (v == b) { present = true; break; }
            if (!present) { bs.push_back(b); cs.push_back(256 + n); ++n; }
        }
        for (size_t i = 0; i < bs.size(); ++i)
            out[(uint32_t)cs[i]] = (uint8_t)bs[i];
        return out;
    }();
    return m;
}

// Decode one UTF-8 codepoint; advances `i`. Returns false on a malformed sequence.
bool next_codepoint(const std::string& s, size_t& i, uint32_t& cp) {
    if (i >= s.size()) return false;
    const unsigned char c = (unsigned char)s[i];
    int extra;
    if (c < 0x80)      { cp = c;          extra = 0; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; extra = 1; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; extra = 2; }
    else if ((c >> 3) == 0x1E){ cp = c & 0x07; extra = 3; }
    else return false;
    if (i + (size_t)extra >= s.size()) return false;
    for (int k = 1; k <= extra; ++k) {
        const unsigned char cc = (unsigned char)s[i + k];
        if ((cc >> 6) != 0x2) return false;
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += (size_t)extra + 1;
    return true;
}

}  // namespace

bool kimi_k3_vocab_from_gguf(const GGUF& g, KimiK3Vocab& out) {
    const std::vector<std::string>* toks = g.meta_str_array("tokenizer.ggml.tokens");
    if (!toks || toks->empty()) {
        std::fprintf(stderr, "[k3] tokenizer.ggml.tokens absent — cannot detokenize\n");
        return false;
    }
    out.tokens = *toks;
    out.eos_id = (int)g.meta_int("tokenizer.ggml.eos_token_id", -1);
    out.bos_id = (int)g.meta_int("tokenizer.ggml.bos_token_id", -1);
    return true;
}

std::string kimi_k3_detokenize_one(const KimiK3Vocab& v, int id) {
    if (id < 0 || id >= v.size()) return std::string();
    const std::string& raw = v.tokens[(size_t)id];
    const auto& dec = byte_decoder();
    std::string out;
    out.reserve(raw.size());
    size_t i = 0;
    uint32_t cp = 0;
    while (next_codepoint(raw, i, cp)) {
        auto it = dec.find(cp);
        if (it != dec.end()) {
            out.push_back((char)it->second);
        } else {
            // Not a byte-level codepoint — an added/special token stored as literal
            // UTF-8 rather than byte-escaped. Re-encode the codepoint unchanged
            // instead of dropping it, so special tokens survive a round trip.
            if (cp < 0x80) out.push_back((char)cp);
            else if (cp < 0x800) {
                out.push_back((char)(0xC0 | (cp >> 6)));
                out.push_back((char)(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                out.push_back((char)(0xE0 | (cp >> 12)));
                out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back((char)(0x80 | (cp & 0x3F)));
            } else {
                out.push_back((char)(0xF0 | (cp >> 18)));
                out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back((char)(0x80 | (cp & 0x3F)));
            }
        }
    }
    return out;
}

std::string kimi_k3_detokenize(const KimiK3Vocab& v, const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) out += kimi_k3_detokenize_one(v, id);
    return out;
}

}  // namespace sparkinfer
