// Backend selection. No CUDA — see the header for why.

#include "sparkinfer/tp/backend_select.h"

#include <sstream>

namespace sparkinfer {
namespace tp {

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::None:        return "none";
        case Backend::Nccl:        return "nccl";
        case Backend::PeerOneShot: return "peer-oneshot";
        case Backend::Multimem:    return "multimem";
    }
    return "?";
}

bool backend_vendored(Backend b) {
    switch (b) {
        case Backend::None: return true;
        case Backend::Nccl: return true;
        // Not in this tree. The implementations exist in the cacheon comm engine
        // but are vLLM-derived (Apache-2.0) and self-declared UNTESTED on
        // hardware; vendoring them is a separate, deliberate decision.
        case Backend::PeerOneShot:
        case Backend::Multimem:
            return false;
    }
    return false;
}

Backend backend_from_string(const std::string& s, std::string* reason) {
    if (s.empty() || s == "nccl")    return Backend::Nccl;
    if (s == "none" || s == "off")   return Backend::None;
    if (s == "peer" || s == "peer-oneshot" || s == "oneshot") return Backend::PeerOneShot;
    if (s == "multimem" || s == "nvls") return Backend::Multimem;
    if (reason) {
        std::ostringstream os;
        os << "unknown SPARKINFER_TP_BACKEND '" << s << "' — using nccl "
              "(valid: nccl, peer, multimem, none)";
        *reason = os.str();
    }
    return Backend::Nccl;
}

Backend select_backend(Backend requested, int tp_size, const Capabilities& caps,
                       std::string* reason) {
    auto say = [&](const std::string& m) { if (reason) *reason = m; };

    // TP=1 issues no collective at all, whatever was requested. This is the path
    // that keeps the default single-GPU build bit-identical to the pre-TP one.
    if (tp_size <= 1) {
        if (requested != Backend::None && requested != Backend::Nccl) {
            std::ostringstream os;
            os << "tp_size=" << tp_size << ": no collective needed, ignoring "
               << backend_name(requested);
            say(os.str());
        }
        return Backend::None;
    }

    // Cannot form the group. Downgrading to None here would silently run an
    // unsharded model, so this is reported as the hard error it is — the caller
    // is expected to refuse to continue.
    if (caps.device_count < tp_size) {
        std::ostringstream os;
        os << "tp_size=" << tp_size << " but only " << caps.device_count
           << " device(s) visible — cannot form the TP group";
        say(os.str());
        return Backend::None;
    }

    if (requested == Backend::None) {
        std::ostringstream os;
        os << "tp_size=" << tp_size << " with backend=none: ranks would never "
              "exchange partial sums, so every rank would emit a partial result";
        say(os.str());
        return Backend::None;
    }

    // Hardware gates first, so the message names the real reason rather than
    // "not vendored" when the box could not have run it anyway.
    if (requested == Backend::Multimem) {
        if (!caps.multicast_supported) {
            say("multimem requested but CUDA multicast is unsupported here — using nccl");
            return Backend::Nccl;
        }
        if (caps.min_compute_capability < 90) {
            std::ostringstream os;
            os << "multimem needs sm_90+, box is sm_" << caps.min_compute_capability
               << " — using nccl";
            say(os.str());
            return Backend::Nccl;
        }
    }
    if (requested == Backend::PeerOneShot && !caps.peer_access_all_pairs) {
        say("peer-oneshot needs peer access across all pairs — using nccl");
        return Backend::Nccl;
    }

    if (!backend_vendored(requested)) {
        std::ostringstream os;
        os << backend_name(requested) << " is not implemented in this tree yet "
              "(vLLM-derived, Apache-2.0, and untested on hardware) — using nccl";
        say(os.str());
        return Backend::Nccl;
    }

    if (requested == Backend::Nccl && !caps.nccl_available) {
        say("nccl requested but NCCL is not available (build with -DSPARKINFER_TP=ON "
            "and install NCCL) — no collective; refuse to run sharded");
        return Backend::None;
    }

    return requested;
}

}  // namespace tp
}  // namespace sparkinfer
