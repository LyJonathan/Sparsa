// include/strata/minimax/ram_plan.hpp - the pinned RAM tier's budget and the headroom the automatic plan keeps, as
// pure functions so the arithmetic can be unit-tested without a machine's memory state.
//
// What the old automatic plan got wrong (measured on the 146 GB MiniMax pack, 128 GB box): it sized the tier from the
// AVAILABLE RAM minus `ram_headroom_auto` = physical/16 clamped to [3,10] GiB = 8 GiB, and then kept only HALF of that
// headroom free (`avail - headroom/2` = avail - 4 GiB).  Neither the 10 GB floor nor the working set the run grows
// ABOVE the tier was in the sum:
//
//   tier 84.0 GiB planned with 96 GiB available  ->  peak working set 90.6 GiB  ->  6.5 GiB left
//
// The 6.6 GiB above the tier is the residency of the weights' mapping plus the token workspace and the combine
// buffers; it is a property of the run, not of the tier, so it belongs in the headroom.  The rules here are
//     headroom = max(10 GiB, physical/12) + over_tier      (over_tier defaults to the measured 7 GiB)
//     tier    <= min(physical - headroom - dense/KV/work overhead, available - headroom)
// so that `available after the fill` stays at or above the 10 GB floor.
#ifndef STRATA_MINIMAX_RAM_PLAN_HPP
#define STRATA_MINIMAX_RAM_PLAN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::minimax {

/// The measured working set a MiniMax run grows above the pinned tier (mmap residency of the weights, the token
/// workspace, the combine buffers): 6.6 GiB in the 146 GB pack's plan, rounded up.
inline constexpr size_t kRamOverTierBytes = (size_t)7 << 30;

/// The 10 GB floor the shared-PC launcher enforces; the automatic plan must respect it on its own.
inline constexpr size_t kRamFloorBytes = (size_t)10 << 30;

/// The headroom the automatic plan keeps: `max(10 GiB, physical/12) + over_tier`.
inline size_t ram_headroom_auto(size_t physical_bytes, size_t over_tier = kRamOverTierBytes) {
    const size_t base = std::max<size_t>(physical_bytes / 12, kRamFloorBytes);
    return base + over_tier;
}

/// The pinned tier's budget. `overhead_bytes` is the host-side cost the plan already knows (dense host copies, host
/// KV, work buffers); the headroom carries the floor and the measured over-tier working set.
inline size_t ram_tier_budget(size_t physical_bytes, size_t available_bytes, size_t headroom_bytes,
                              size_t overhead_bytes) {
    const size_t room = physical_bytes > headroom_bytes + overhead_bytes
                            ? physical_bytes - headroom_bytes - overhead_bytes
                            : 0;
    const size_t free_now = available_bytes > headroom_bytes ? available_bytes - headroom_bytes : 0;
    return std::min(room, free_now);
}

}  // namespace strata::minimax
#endif  // STRATA_MINIMAX_RAM_PLAN_HPP
