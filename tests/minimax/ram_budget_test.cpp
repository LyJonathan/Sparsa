// tests/minimax/ram_budget_test.cpp - what the automatic pinned-tier budget keeps free, with no GPU.
//
// The bug this guards: the budget was min(total - ..., available - 2 x headroom - prompt).  The CPU-resident dense
// weights were not in the available-RAM term, and on Windows their pages come back into the working set after the
// snapshot, so a tier sized from the snapshot could leave less free than the watchdog floor (u2: 74.2 GiB tier, 12.8 GiB
// host dense, 9.8 GiB available).  The rule is now ram_keep_bytes(); these checks pin the numbers it has to produce:
//   * on a 64-160 GiB box the margin is still 2 x headroom, as before (the Linux rig-2 at 121 GB and the 128 GB PC);
//     the 8 GiB floor and the tenth of physical RAM only raise it on smaller or larger boxes;
//   * the Windows dense term adds exactly the mapped bytes, and Linux adds nothing;
//   * an explicit --minimax-ram-headroom-mib is used as given.
#include "strata/minimax/ram_budget.hpp"

#include <cstdio>
#include <cstdint>

using namespace strata::minimax;

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// The engine's automatic headroom: physical / 16 clamped to 3-10 GiB (ram_headroom_auto in engine.cpp).
static size_t auto_headroom(size_t phys) {
    const size_t GiB = (size_t)1 << 30;
    size_t v = phys / 16;
    if (v < 3 * GiB) v = 3 * GiB;
    if (v > 10 * GiB) v = 10 * GiB;
    return v;
}

int main() {
    const size_t MiB = (size_t)1 << 20;
    const size_t GiB = (size_t)1 << 30;

    // 1. Where the old keep was already the binding margin, the new keep is the old one, byte for byte (no mapped term).
    for (size_t gib : {64, 96, 121, 128}) {
        const size_t phys = gib * GiB;
        const size_t hd = auto_headroom(phys);
        const size_t post = 2800 * MiB;   // the batched-prompt pools of a 4096-token prompt on two cards, about
        const size_t old_keep = 2 * hd + post;
        const size_t new_keep = ram_keep_bytes(phys, hd, true, post, 0);
        expect(new_keep == old_keep, "a Linux-style keep (no mapped term) equals 2 x headroom + prompt from 64 to 128 GiB");
    }

    // 2. The 128 GB Windows PC, default flags, the measured case: headroom 8 GiB, prompt pools 2.7 GiB, host dense 4.4
    //    GiB of which 2.5 GiB is the embedding (gathered, not resident).  Keep = 16 + 2.7 + 1.9 GiB.
    {
        const size_t phys = 128 * GiB;
        const size_t hd = auto_headroom(phys);
        expect(hd == 8 * GiB, "128 GiB box: automatic headroom is 8 GiB");
        const size_t keep = ram_keep_bytes(phys, hd, true, 2700 * MiB, 1900 * MiB);
        expect(keep == 16 * GiB + 2700 * MiB + 1900 * MiB, "PC: keep = 2 x headroom + prompt + mapped dense");
    }

    // 3. The overrun case: 12.8 GiB of host dense, 10.3 GiB of it resident.  The keep rises by the mapped part.
    {
        const size_t phys = 128 * GiB;
        const size_t keep = ram_keep_bytes(phys, 8 * GiB, true, 2800 * MiB, 10300 * MiB);
        expect(keep == 16 * GiB + 2800 * MiB + 10300 * MiB, "dense-on-host box: keep includes the resident dense");
        expect(keep > ram_keep_bytes(phys, 8 * GiB, true, 2800 * MiB, 0), "the mapped term only ever adds");
    }

    // 4. Small and large boxes: the 8 GiB floor and the tenth of physical RAM.
    {
        const size_t small = 32 * GiB;
        expect(ram_keep_bytes(small, auto_headroom(small), true, 0, 0) == 8 * GiB,
               "32 GiB box: margin is the 8 GiB floor (6 GiB before)");
        const size_t large = 256 * GiB;
        expect(ram_keep_bytes(large, auto_headroom(large), true, 0, 0) == 256 * GiB / 10,
               "256 GiB box: margin is a tenth of physical RAM (20 GiB before)");
    }

    // 5. An explicit headroom is used as given: no floor is added to it.
    {
        const size_t phys = 128 * GiB;
        expect(ram_keep_bytes(phys, 2 * GiB, false, 0, 0) == 4 * GiB,
               "explicit headroom 2 GiB: margin is 2 x headroom, no floor");
    }

    // 6. The no-tier path (no pinned expert tier, so the batched prompt's pools are the only host pools it sizes).
    //    128 GiB, automatic headroom 8 GiB (margin 16 GiB), 2.8 GiB of fixed pools, 100 KiB a token.
    {
        const size_t phys = 128 * GiB, hd = 8 * GiB, fixed = 2800 * MiB, per_tok = 100 * 1024;
        expect(ram_prompt_tokens_fit(phys, hd, true, 30 * GiB, fixed, per_tok, 0, 4096) == 4096,
               "no-tier: a 4096-token chunk is kept whole when 30 GiB are available");
        const size_t tight = ram_prompt_tokens_fit(phys, hd, true, 19 * GiB, fixed, per_tok, 0, 4096);
        expect(tight >= 2 && tight < 4096, "no-tier: 19 GiB available shrinks the chunk below 4096");
        expect(ram_prompt_tokens_fit(phys, hd, true, 17 * GiB, fixed, per_tok, 0, 4096) == 0,
               "no-tier: 17 GiB available (below margin + pools) keeps the token path (0)");
        expect(ram_prompt_tokens_fit(phys, hd, true, 30 * GiB, fixed, per_tok, 0, 1) == 1,
               "no-tier: a requested chunk below 2 is returned unchanged");
        // Whatever chunk is returned, the keep rule still holds: margin + pools + mapped <= available.
        for (size_t avail_gib : {17, 18, 19, 20, 21, 22, 24}) {
            for (size_t mapped : {(size_t)0, 1900 * MiB}) {
                const size_t avail = avail_gib * GiB;
                const size_t T = ram_prompt_tokens_fit(phys, hd, true, avail, fixed, per_tok, mapped, 4096);
                const size_t keep = ram_keep_bytes(phys, hd, true, fixed + T * per_tok, mapped);
                expect(T == 0 || keep <= avail, "no-tier: the returned chunk leaves the budget's keep free");
                expect(T < 2 || ram_keep_bytes(phys, hd, true, fixed + (T + 1) * per_tok, mapped) > avail || T == 4096,
                       "no-tier: the returned chunk is the largest that fits");
            }
        }
        // The Windows dense term is charged too: 1.9 GiB of resident dense takes the room that 19 GiB left.
        expect(ram_prompt_tokens_fit(phys, hd, true, 19 * GiB, fixed, per_tok, 1900 * MiB, 4096) == 0,
               "no-tier: the mapped dense term counts against the chunk (Windows)");
    }
    // 7. An explicit headroom, and a zero per-token cost: the fit is the requested chunk, or the floor's keep only.
    {
        const size_t phys = 128 * GiB;
        expect(ram_prompt_tokens_fit(phys, 2 * GiB, false, 6 * GiB, 1 * GiB, 1 * MiB, 0, 4096) == 1024,
               "no-tier, explicit 2 GiB headroom: 6 GiB - 4 GiB margin - 1 GiB pools = 1024 MiB of 1 MiB tokens");
        expect(ram_prompt_tokens_fit(phys, 2 * GiB, false, 6 * GiB, 1 * GiB, 0, 0, 4096) == 4096,
               "no-tier: zero per-token cost keeps the requested chunk");
    }

    if (failures) {
        std::fprintf(stderr, "ram_budget_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("ram_budget_test: ok\n");
    return 0;
}
