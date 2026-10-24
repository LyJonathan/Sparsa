// tests/minimax/vram_plan_test.cpp - the automatic VRAM-slot sizing, with no GPU.
//
// The bug this guards: the old auto sizing used `free - reserve` and never subtracted the batched-prefill workspace
// that init_prefill allocates later, so on a card large enough for the slots to fill it (a 96 GB card running a
// 153 GB model) the prefill allocation died in cudaMalloc at startup.  The rule is now an invariant:
//     reserve + prefill + slots * moe_layers * expert_bytes <= free
// and the explicit slot count still only has to cover its own bytes (the prefill batch shrinks to fit instead).
#include "strata/minimax/vram_plan.hpp"

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

int main() {
    const size_t MiB = (size_t)1 << 20;
    const size_t GiB = (size_t)1 << 30;
    // MiniMax-M3.1-Flash's real numbers: one expert is 17.4 MB, 57 MoE layers, 128 experts.
    const size_t expert = 17'400'000;
    const int moe = 57;
    const int nexp = 128;

    {
        // The 96 GB card that used to OOM: free VRAM after dense/KV/workspace, a 0.5 GiB prefill workspace.  The free
        // figure is chosen so the old rule took the last slot and left half the prefill workspace unbacked.
        const size_t reserve = 700 * MiB;
        const size_t prefill = 512 * MiB;
        const size_t per = expert * (size_t)moe;
        const size_t free_bytes = reserve + 100 * per + prefill / 2;
        const VramSlotPlan p = plan_vram_slots(free_bytes, reserve, prefill, expert, moe, nexp, -1);
        expect(p.prefill_reserved, "auto plan marks the prefill reserve");
        expect(p.slots > 0, "auto plan still uses the card");
        expect(reserve + prefill + p.slot_bytes <= free_bytes, "auto plan leaves room for the prefill workspace");
        const size_t old = (free_bytes - reserve) / per;
        expect(old == 100, "the buggy rule would take 100 slots");
        expect((size_t)p.slots < old, "auto plan is smaller than the buggy one on a big card");
    }
    {
        // A 12 GB card: the same invariant, and the prefill reserve costs at most one slot per layer.
        const size_t free_bytes = 11 * GiB;
        const size_t reserve = 700 * MiB;
        const size_t prefill = 400 * MiB;
        const VramSlotPlan p = plan_vram_slots(free_bytes, reserve, prefill, expert, 28, nexp, -1);
        const size_t old = (free_bytes - reserve) / (expert * (size_t)28);
        expect(reserve + prefill + p.slot_bytes <= free_bytes, "12 GB plan leaves room for the prefill workspace");
        expect(old - (size_t)p.slots <= 1, "the prefill reserve costs at most one slot on a 12 GB card");
    }
    {
        // No prefill: identical to the old rule, so the auto plan on a box that keeps prefill off does not move.
        const size_t free_bytes = 11 * GiB;
        const size_t reserve = 700 * MiB;
        const VramSlotPlan p = plan_vram_slots(free_bytes, reserve, 0, expert, 28, nexp, -1);
        const size_t old = (free_bytes - reserve) / (expert * (size_t)28);
        expect((size_t)p.slots == old, "no-prefill auto plan is unchanged");
        expect(!p.prefill_reserved, "no prefill to reserve");
    }
    {
        // An explicit count is honoured up to n_expert and no further.
        const VramSlotPlan p = plan_vram_slots(11 * GiB, 700 * MiB, 0, expert, 28, nexp, 200);
        expect(p.slots == nexp, "explicit count clamps to n_expert");
        const VramSlotPlan z = plan_vram_slots(11 * GiB, 700 * MiB, 0, expert, 28, nexp, 0);
        expect(z.slots == 0 && z.slot_bytes == 0, "explicit zero keeps no slots");
    }
    {
        // Degenerate inputs do not underflow or divide by zero.
        const VramSlotPlan p = plan_vram_slots(0, 700 * MiB, 512 * MiB, expert, moe, nexp, -1);
        expect(p.slots == 0 && p.slot_bytes == 0, "no free VRAM yields no slots");
        const VramSlotPlan n = plan_vram_slots(1 * GiB, 0, 0, expert, 0, nexp, -1);
        expect(n.slots == 0, "a device with no MoE layers yields no slots");
        const VramSlotPlan huge = plan_vram_slots(100 * MiB, 0, 0, expert, moe, nexp, -1);
        expect(huge.slots == 0, "an expert larger than the budget yields no slots");
    }
    if (failures == 0) std::printf("vram_plan_test: ok\n");
    return failures == 0 ? 0 : 1;
}
