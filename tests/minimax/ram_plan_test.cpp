// tests/minimax/ram_plan_test.cpp - the pinned RAM tier's automatic budget, with no machine state.
//
// The bug this guards (measured on the 146 GB MiniMax pack, 128 GB box): the automatic plan kept only half of an 8 GiB
// headroom (avail - 4 GiB) and ignored both the 10 GB floor and the 6.6 GiB the run grows above the tier, so it
// planned an 84 GiB tier with 96 GiB available and left 6.5 GiB - under the floor the shared-PC launcher enforces.
#include "strata/minimax/ram_plan.hpp"

#include <cstdio>

using namespace strata::minimax;

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    const size_t gib = (size_t)1 << 30;

    // 1. the headroom always carries the floor, and the over-tier working set on top of it
    expect(ram_headroom_auto(128 * gib) >= kRamFloorBytes + kRamOverTierBytes, "128 GiB box: floor + over-tier");
    expect(ram_headroom_auto(32 * gib) >= kRamFloorBytes + kRamOverTierBytes, "32 GiB box: the floor is a floor");
    expect(ram_headroom_auto(128 * gib) == 128 * gib / 12 + kRamOverTierBytes, "128 GiB: physical/12 + over-tier");
    expect(ram_headroom_auto(1 * gib) == kRamFloorBytes + kRamOverTierBytes, "a tiny box still keeps the floor");

    // 2. the plan leaves at least the floor free, for the case that failed (96 GiB available on a 128 GiB box)
    {
        const size_t phys = 128 * gib, avail = 96 * gib, overhead = 4 * gib;
        const size_t head = ram_headroom_auto(phys);
        const size_t tier = ram_tier_budget(phys, avail, head, overhead);
        expect(tier <= avail - head, "96 GiB available: the tier keeps the whole headroom free");
        expect(avail - tier >= kRamFloorBytes, "96 GiB available: at least the 10 GB floor survives");
        expect(tier < 96 * gib, "96 GiB available: the tier is not the whole free RAM");
    }

    // 3. the old rule would have failed the same case: half a headroom is not a floor
    {
        const size_t phys = 128 * gib, avail = 96 * gib;
        const size_t old_head = phys / 16;                      // 8 GiB
        const size_t old_tier = avail - old_head / 2;            // what the engine used to do
        expect(avail - old_tier < kRamFloorBytes, "the old rule left under the floor (so this test would catch it)");
    }

    // 4. the physical bound: never plan more than physical - headroom - overhead
    {
        const size_t phys = 128 * gib, head = ram_headroom_auto(phys), overhead = 4 * gib;
        const size_t tier = ram_tier_budget(phys, 500 * gib, head, overhead);
        expect(tier == phys - head - overhead, "a huge machine does not plan past physical - headroom - overhead");
    }

    // 5. an idle box with plenty free: the available side binds, not the physical side
    {
        const size_t phys = 128 * gib, avail = 120 * gib, overhead = 4 * gib;
        const size_t head = ram_headroom_auto(phys);
        const size_t tier = ram_tier_budget(phys, avail, head, overhead);
        expect(tier == std::min(phys - head - overhead, avail - head), "the tighter of the two bounds wins");
        expect(avail - tier >= kRamFloorBytes + kRamOverTierBytes - kRamOverTierBytes, "floor survives");
    }

    // 6. degenerate inputs do not wrap
    expect(ram_tier_budget(0, 0, 0, 0) == 0, "all zero: zero");
    expect(ram_tier_budget(1 * gib, 1 * gib, 4 * gib, 0) == 0, "headroom above physical: nothing to pin");
    expect(ram_tier_budget(8 * gib, 4 * gib, 8 * gib, 0) == 0, "available under the headroom: nothing to pin");

    std::printf("%s\n", failures == 0 ? "ram_plan_test: PASS" : "ram_plan_test: FAIL");
    return failures == 0 ? 0 : 1;
}
