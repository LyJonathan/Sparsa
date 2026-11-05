// tests/minimax/ram_budget_test.cpp - what the automatic pinned-tier budget keeps free, with no GPU.
//
// The bug this guards: the budget was min(total - ..., available - 2 x headroom - prompt).  The CPU-resident dense
// weights were not in the available-RAM term, and on Windows their pages come back into the working set after the
// snapshot, so a tier sized from the snapshot could leave less free than the watchdog floor (u2: 74.2 GiB tier, 12.8 GiB
// host dense, 9.8 GiB available).  The rule is now ram_keep_bytes(); these checks pin the numbers it has to produce:
//   * on a 64-160 GiB box the margin is still 2 x headroom, as before (the Linux rig-2 at 121 GB and the 128 GB PC);
//     the 8 GiB floor and the tenth of physical RAM only raise it on smaller or larger boxes;
//   * the Windows dense term adds exactly the mapped bytes, and Linux adds nothing;
//   * an explicit --minimax-ram-headroom-mib is used as given;
//   * the frequency walk (ram_slots_by_profile) never reserves more bytes than the uniform plan (cases 8-11).
//   * the automatic pinned tier (ram_auto_tier_bytes) follows the RAM the box has, is capped at 108 GiB and keeps the
//     10 GiB floor free on every box from 16 to 256 GiB (case 12).
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

    // 8. The frequency walk (ram_slots_by_profile) never reserves more bytes than the uniform plan.  Three MoE layers of
    //    1, 1 and 2 MiB experts, budget 10 MiB, cap 4: the uniform plan is floor(10 / (3 x 2)) = 1 expert a layer, 4 MiB in
    //    all.  The walk used to fill the whole 10 MiB; it must stop at 4.  The hottest entries come first.
    {
        const std::vector<char> moe = {1, 1, 1};
        const std::vector<int> vram = {0, 0, 0};
        const std::vector<size_t> xb = {1 * MiB, 1 * MiB, 2 * MiB};
        const std::vector<std::pair<int32_t, int32_t>> prof = {{0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}};
        const std::vector<int> ram = ram_slots_by_profile(prof, 0, moe, vram, xb, 4, 10 * MiB);
        size_t bytes = 0;
        for (size_t i = 0; i < ram.size(); ++i) bytes += (size_t)ram[i] * xb[i];
        expect(bytes <= 4 * MiB, "frequency walk: no more bytes than the uniform plan (4 MiB, not the 10 MiB budget)");
        expect(ram[0] == 3 && ram[1] == 1 && ram[2] == 0,
               "frequency walk: the hottest entries come first and the round-robin fills what the cap leaves");
    }
    // 9. VRAM slots are skipped, non-MoE layers get nothing, and the profile's layer 0 is absolute layer 2 here.
    {
        const std::vector<char> moe = {0, 0, 1, 1, 1};
        const std::vector<int> vram = {0, 0, 1, 0, 0};
        const std::vector<size_t> xb = {0, 0, 1 * MiB, 1 * MiB, 1 * MiB};
        const std::vector<std::pair<int32_t, int32_t>> prof = {{0, 0}, {0, 1}, {1, 0}, {2, 0}};
        const std::vector<int> ram = ram_slots_by_profile(prof, 2, moe, vram, xb, 4, 3 * MiB);
        expect(ram[0] == 0 && ram[1] == 0, "frequency walk: non-MoE layers get no RAM slots");
        expect(ram[2] == 1 && ram[3] == 1 && ram[4] == 1,
               "frequency walk: a layer's first profile entry is its VRAM slot, the next ones go to RAM");
    }
    // 10. With budget to spare the cap does not bind: every layer reaches the expert cap, as the walk always did.
    {
        const std::vector<char> moe = {1, 1, 1};
        const std::vector<int> vram = {0, 0, 0};
        const std::vector<size_t> xb = {1 * MiB, 1 * MiB, 2 * MiB};
        std::vector<std::pair<int32_t, int32_t>> prof;
        for (int32_t l = 0; l < 3; ++l)
            for (int32_t e = 0; e < 4; ++e) prof.push_back({l, e});
        const std::vector<int> ram = ram_slots_by_profile(prof, 0, moe, vram, xb, 4, 1000 * MiB);
        expect(ram[0] == 4 && ram[1] == 4 && ram[2] == 4,
               "frequency walk: with budget to spare every layer reaches the cap (no regression)");
    }
    // 11. For every budget from 1 to 40 MiB the walk stays inside the uniform plan's bytes and inside the cap.  Layers:
    //     MoE 3 MiB, non-MoE, MoE 1 MiB, MoE 2 MiB; VRAM slots 1, 0, 0, 2; cap 6.
    {
        const std::vector<char> moe = {1, 0, 1, 1};
        const std::vector<int> vram = {1, 0, 0, 2};
        const std::vector<size_t> xb = {3 * MiB, 0, 1 * MiB, 2 * MiB};
        std::vector<std::pair<int32_t, int32_t>> prof;
        for (int32_t l = 0; l < 3; ++l)
            for (int32_t e = 0; e < 8; ++e) prof.push_back({l, e});
        for (size_t mib = 1; mib <= 40; ++mib) {
            const std::vector<int> ram = ram_slots_by_profile(prof, 0, moe, vram, xb, 6, mib * MiB);
            const size_t uniform = std::min<size_t>(mib * MiB / (3 * 3 * MiB), 6);
            const size_t uniform_bytes = uniform * (3 * MiB + 1 * MiB + 2 * MiB);
            size_t bytes = 0;
            bool within_cap = true;
            for (size_t i = 0; i < ram.size(); ++i) {
                bytes += (size_t)ram[i] * xb[i];
                if (ram[i] > 6) within_cap = false;
            }
            expect(bytes <= uniform_bytes, "frequency walk: bytes stay within the uniform plan at every budget from 1 to 40 MiB");
            expect(within_cap, "frequency walk: no layer passes the expert cap");
            expect(ram[1] == 0, "frequency walk: the non-MoE layer stays at zero");
        }
    }

    // 12. The automatic pinned tier follows the RAM the box has.  It is the smallest of the room, what the box keeps free
    //     (at least the 10 GiB floor) and the 108 GiB ceiling; no box goes below the floor, and no size is hard-coded.
    {
        const size_t kCeil = (size_t)108 * GiB, kFloor = (size_t)10 * GiB;
        const size_t post = 2700 * MiB;   // the batched prompt's pools on two cards, as in case 2
        for (size_t gib : {16, 32, 64, 96, 128, 256}) {
            const size_t phys = gib * GiB;
            const size_t hd = auto_headroom(phys);
            const size_t room = phys - hd;
            for (size_t mapped : {(size_t)0, (size_t)1900 * MiB}) {
                const size_t keep = ram_keep_bytes(phys, hd, true, post, mapped);
                size_t prev = 0;
                for (size_t pct : {10, 25, 50, 75, 90, 100}) {
                    const size_t avail = phys / 100 * pct;
                    const size_t tier = ram_auto_tier_bytes(room, avail, keep);
                    expect(tier <= kCeil, "auto tier: never above the 108 GiB ceiling");
                    expect(tier <= room, "auto tier: never above the room the box leaves");
                    expect(tier == 0 || avail - tier >= std::max(keep, kFloor),
                           "auto tier: leaves at least its keep and the 10 GiB floor free");
                    expect(tier >= prev, "auto tier: does not shrink as the available RAM grows");
                    prev = tier;
                }
            }
        }
        // The 128 GB PC at idle (124 GiB available, the measured keep of 20.6 GiB): the tier is what is left, under the cap.
        {
            const size_t phys = 128 * GiB, hd = auto_headroom(phys);
            const size_t keep = ram_keep_bytes(phys, hd, true, post, 1900 * MiB);
            const size_t tier = ram_auto_tier_bytes(phys - hd, 124 * GiB, keep);
            expect(tier == 124 * GiB - keep, "128 GiB PC at idle: the tier is the available RAM less the keep");
            expect(tier < kCeil, "128 GiB PC at idle: the tier stays under the 108 GiB ceiling");
        }
        // A 256 GB box with the RAM free: the ceiling binds, so the tier is exactly 108 GiB.
        expect(ram_auto_tier_bytes(256 * GiB - auto_headroom(256 * GiB), 250 * GiB,
                                   ram_keep_bytes(256 * GiB, auto_headroom(256 * GiB), true, post, 0)) == kCeil,
               "256 GiB box with the RAM free: the tier is the 108 GiB ceiling");
        // A 64 GB box with 60 GiB free takes what is left above its keep (no size is fixed).
        {
            const size_t phys = 64 * GiB, hd = auto_headroom(phys);
            const size_t keep = ram_keep_bytes(phys, hd, true, post, 0);
            expect(ram_auto_tier_bytes(phys - hd, 60 * GiB, keep) == 60 * GiB - keep,
                   "64 GiB box with 60 GiB available: the tier is the available RAM less the keep");
        }
        // A 16 GB box with 13 GiB available keeps the 10 GiB floor (its margin is 8 GiB, its prompt 2.7 GiB).
        expect(ram_auto_tier_bytes(16 * GiB - auto_headroom(16 * GiB), 13 * GiB,
                                   ram_keep_bytes(16 * GiB, auto_headroom(16 * GiB), true, post, 0)) == 13 * GiB - ram_keep_bytes(16 * GiB, auto_headroom(16 * GiB), true, post, 0),
               "16 GiB box: the tier leaves the 10 GiB floor free");
        expect(ram_auto_tier_bytes(32 * GiB, 10 * GiB + 512 * MiB, ram_keep_bytes(32 * GiB, auto_headroom(32 * GiB), true, post, 0)) == 0,
               "32 GiB box with less than its keep free: no tier");
    }

    if (failures) {
        std::fprintf(stderr, "ram_budget_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("ram_budget_test: ok\n");
    return 0;
}
