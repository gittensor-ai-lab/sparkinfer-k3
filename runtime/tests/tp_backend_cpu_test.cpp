// CPU-only tests for TP all-reduce backend selection (no GPU needed).
//
// Why this is worth testing rather than eyeballing: if a run silently falls back
// to a different collective than the one an operator asked for, a benchmark
// attributes its number to the wrong mechanism. The comm "optimization" gets
// credited for a change it never made, and the frontier moves on a fiction.
// Selection is pure, so it can be pinned exactly.
//
// Build/run:
//   g++ -std=c++17 -I runtime/include runtime/tests/tp_backend_cpu_test.cpp
//       runtime/src/tp/backend_select.cpp -o /tmp/t && /tmp/t

#include "sparkinfer/tp/backend_select.h"

#include <cstdio>
#include <string>

using namespace sparkinfer::tp;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            std::printf("  FAIL %s:%d: %s\n        ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define CHECK_BACKEND(got, want)                                                \
    CHECK((got) == (want), "got %s want %s", backend_name(got), backend_name(want))

// An 8x H200 HGX node: NVSwitch, all-pairs peer access, multicast, sm_90.
static Capabilities h200x8() {
    Capabilities c;
    c.device_count = 8;
    c.nccl_available = true;
    c.peer_access_all_pairs = true;
    c.multicast_supported = true;
    c.min_compute_capability = 90;
    return c;
}

// A single consumer card: no peers, no multicast, sm_120.
static Capabilities rtx5090() {
    Capabilities c;
    c.device_count = 1;
    c.nccl_available = true;
    c.peer_access_all_pairs = false;
    c.multicast_supported = false;
    c.min_compute_capability = 120;
    return c;
}

static void test_tp1_never_issues_a_collective() {
    std::printf("tp1_never_issues_a_collective\n");
    // The equivalence that makes it safe to land TP before validating it: at
    // TP=1 nothing is exchanged, so the forward is bit-identical to the pre-TP path.
    std::string why;
    CHECK_BACKEND(select_backend(Backend::Nccl, 1, h200x8(), &why), Backend::None);
    CHECK_BACKEND(select_backend(Backend::Multimem, 1, h200x8(), &why), Backend::None);
    CHECK(!why.empty(), "downgrade from an explicit request should be explained");
    CHECK_BACKEND(select_backend(Backend::Nccl, 0, h200x8(), &why), Backend::None);
    CHECK_BACKEND(select_backend(Backend::Nccl, -1, h200x8(), &why), Backend::None);
}

static void test_nccl_is_the_default_on_a_real_node() {
    std::printf("nccl_is_the_default_on_a_real_node\n");
    std::string why;
    CHECK_BACKEND(select_backend(Backend::Nccl, 8, h200x8(), &why), Backend::Nccl);
    CHECK(why.empty(), "no downgrade, so no reason expected: %s", why.c_str());
}

static void test_unvendored_backends_fall_back_and_say_so() {
    std::printf("unvendored_backends_fall_back_and_say_so\n");
    // peer-oneshot and multimem are declared but not in this tree. They must
    // never appear to have run — a benchmark naming them would be measuring NCCL.
    CHECK(!backend_vendored(Backend::PeerOneShot), "peer-oneshot must not claim to be vendored");
    CHECK(!backend_vendored(Backend::Multimem), "multimem must not claim to be vendored");
    CHECK(backend_vendored(Backend::Nccl), "nccl is vendored");

    for (Backend b : {Backend::PeerOneShot, Backend::Multimem}) {
        std::string why;
        CHECK_BACKEND(select_backend(b, 8, h200x8(), &why), Backend::Nccl);
        CHECK(why.find("not implemented") != std::string::npos,
              "%s downgrade should say it is not implemented, got: %s",
              backend_name(b), why.c_str());
    }
}

static void test_hardware_gates_are_reported_before_vendoring() {
    std::printf("hardware_gates_are_reported_before_vendoring\n");
    // When the box could not have run it anyway, say THAT rather than
    // "not vendored" — otherwise vendoring looks like the fix when it is not.
    Capabilities no_mc = h200x8();
    no_mc.multicast_supported = false;
    std::string why;
    CHECK_BACKEND(select_backend(Backend::Multimem, 8, no_mc, &why), Backend::Nccl);
    CHECK(why.find("multicast") != std::string::npos, "should name multicast: %s", why.c_str());

    Capabilities old_arch = h200x8();
    old_arch.min_compute_capability = 89;      // pre-Hopper
    why.clear();
    CHECK_BACKEND(select_backend(Backend::Multimem, 8, old_arch, &why), Backend::Nccl);
    CHECK(why.find("sm_90") != std::string::npos, "should name sm_90: %s", why.c_str());

    Capabilities no_peer = h200x8();
    no_peer.peer_access_all_pairs = false;
    why.clear();
    CHECK_BACKEND(select_backend(Backend::PeerOneShot, 8, no_peer, &why), Backend::Nccl);
    CHECK(why.find("peer access") != std::string::npos, "should name peer access: %s", why.c_str());
}

static void test_too_few_devices_is_a_hard_downgrade() {
    std::printf("too_few_devices_is_a_hard_downgrade\n");
    // Asking for TP=8 on one card must NOT quietly become a working TP=1 run:
    // the caller has to refuse, or it silently benchmarks a different config
    // than the one it reports.
    std::string why;
    CHECK_BACKEND(select_backend(Backend::Nccl, 8, rtx5090(), &why), Backend::None);
    CHECK(why.find("cannot form") != std::string::npos,
          "should say the group cannot be formed: %s", why.c_str());
    CHECK(why.find("only 1") != std::string::npos, "should report the count: %s", why.c_str());
}

static void test_backend_none_at_tp8_is_rejected_loudly() {
    std::printf("backend_none_at_tp8_is_rejected_loudly\n");
    // tp_size 8 with no collective is not a fast configuration, it is a wrong
    // one: every rank would emit its own partial sum as if it were the answer.
    std::string why;
    CHECK_BACKEND(select_backend(Backend::None, 8, h200x8(), &why), Backend::None);
    CHECK(why.find("partial") != std::string::npos,
          "should explain that partial sums escape: %s", why.c_str());
}

static void test_missing_nccl_does_not_silently_unshard() {
    std::printf("missing_nccl_does_not_silently_unshard\n");
    Capabilities no_nccl = h200x8();
    no_nccl.nccl_available = false;
    std::string why;
    CHECK_BACKEND(select_backend(Backend::Nccl, 8, no_nccl, &why), Backend::None);
    CHECK(why.find("NCCL") != std::string::npos, "should name NCCL: %s", why.c_str());
    CHECK(why.find("refuse") != std::string::npos,
          "should tell the caller to refuse: %s", why.c_str());
}

static void test_env_parsing() {
    std::printf("env_parsing\n");
    std::string why;
    CHECK_BACKEND(backend_from_string("", &why), Backend::Nccl);
    CHECK_BACKEND(backend_from_string("nccl", &why), Backend::Nccl);
    CHECK_BACKEND(backend_from_string("none", &why), Backend::None);
    CHECK_BACKEND(backend_from_string("off", &why), Backend::None);
    CHECK_BACKEND(backend_from_string("peer", &why), Backend::PeerOneShot);
    CHECK_BACKEND(backend_from_string("oneshot", &why), Backend::PeerOneShot);
    CHECK_BACKEND(backend_from_string("multimem", &why), Backend::Multimem);
    CHECK_BACKEND(backend_from_string("nvls", &why), Backend::Multimem);
    CHECK(why.empty(), "known values need no explanation: %s", why.c_str());

    // A typo must not kill a run that took 20 minutes to load 802 GiB.
    why.clear();
    CHECK_BACKEND(backend_from_string("mutlimem", &why), Backend::Nccl);
    CHECK(why.find("unknown") != std::string::npos, "should flag the typo: %s", why.c_str());
    CHECK(why.find("mutlimem") != std::string::npos, "should echo it: %s", why.c_str());
}

static void test_backend_names_are_stable() {
    std::printf("backend_names_are_stable\n");
    // These strings land in result JSON and in reference.lock provenance, so
    // renaming one silently invalidates comparisons against older runs.
    CHECK(std::string(backend_name(Backend::None)) == "none", "none");
    CHECK(std::string(backend_name(Backend::Nccl)) == "nccl", "nccl");
    CHECK(std::string(backend_name(Backend::PeerOneShot)) == "peer-oneshot", "peer-oneshot");
    CHECK(std::string(backend_name(Backend::Multimem)) == "multimem", "multimem");
}

int main() {
    test_tp1_never_issues_a_collective();
    test_nccl_is_the_default_on_a_real_node();
    test_unvendored_backends_fall_back_and_say_so();
    test_hardware_gates_are_reported_before_vendoring();
    test_too_few_devices_is_a_hard_downgrade();
    test_backend_none_at_tp8_is_rejected_loudly();
    test_missing_nccl_does_not_silently_unshard();
    test_env_parsing();
    test_backend_names_are_stable();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
