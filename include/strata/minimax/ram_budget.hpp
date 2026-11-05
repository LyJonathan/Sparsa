// include/strata/minimax/ram_budget.hpp - the host RAM the automatic pinned-tier budget keeps free, as a pure function.
//
// The engine sizes the pinned expert tier from the RAM available when it plans.  The pinned pages are not reclaimable,
// and everything the process takes or keeps resident after that snapshot has to come out of the same RAM, so the
// budget leaves room for it:
//   * a margin for the operating system and the other programs: two headroom, as it always was, and with an automatic
//     headroom never less than max(8 GiB, 1/10 of physical RAM);
//   * the batched prompt's pinned pools, which init_prefill allocates after the plan is made;
//   * on Windows, the CPU-resident dense weights.  Every token reads them through the GGUF file mapping, so their pages
//     stay in the working set, and AvailPhys does not count a working-set page as available: the working-set trimmer
//     gives those pages back only under pressure, so the snapshot does not show them.  Linux counts clean file pages
//     as MemAvailable (they are reclaimable), so the caller passes 0 there.
// Without a pinned tier (sparsa.py --no-tiers) nothing sizes the expert RAM, so the same keep bounds the batched
// prompt's pools through the chunk size (ram_prompt_tokens_fit).
// The automatic pinned tier itself is ram_auto_tier_bytes(): the room the box leaves, what it keeps free (never less than
// the 10 GiB floor), and the 108 GiB ceiling of the RAM share the GPTQ-v2 pack was measured at.
// The frequency walk that fills the pinned tier under a routing profile (ram_slots_by_profile) is here as well.  It never
// reserves more bytes than the uniform plan, so the watchdog floor sees the same free RAM as the uniform plan does.
// Header-only, so the arithmetic can be unit-tested without a GPU (tests/minimax/ram_budget_test.cpp).
#ifndef STRATA_MINIMAX_RAM_BUDGET_HPP
#define STRATA_MINIMAX_RAM_BUDGET_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::minimax {

/// The bytes the automatic budget must not take from the available RAM.
/// \param phys_bytes      physical RAM (MemTotal / ullTotalPhys)
/// \param headroom_bytes  the headroom in use: physical/16 clamped to 3-10 GiB, or --minimax-ram-headroom-mib
/// \param auto_headroom   true when the headroom was not given on the command line
/// \param post_plan_bytes host bytes allocated after the budget is sized (the batched-prompt pools)
/// \param mapped_bytes    CPU-resident dense bytes read through the file mapping (0 where clean file pages count as
///                        available, i.e. Linux)
inline size_t ram_keep_bytes(size_t phys_bytes, size_t headroom_bytes, bool auto_headroom, size_t post_plan_bytes,
                             size_t mapped_bytes) {
    size_t margin = 2 * headroom_bytes;
    if (auto_headroom) {
        const size_t floor_free = std::max<size_t>((size_t)8 << 30, phys_bytes / 10);
        margin = std::max(margin, floor_free);
    }
    return margin + post_plan_bytes + mapped_bytes;
}

/// The batched prompt's chunk size on the no-tier path.  With no pinned expert tier, nothing sizes the host RAM before the
/// prompt pools are allocated, so the chunk is the knob: the largest chunk (at most `requested`) whose pinned pools
/// still leave ram_keep_bytes() free.  The pools are `fixed_bytes + tokens * per_token_bytes`; `mapped_bytes` is the
/// CPU-resident dense term (Windows only, as in ram_keep_bytes).  Returns `requested` unchanged when it is below 2 (no
/// batch to shrink), 0 when not even a 2-token chunk fits (the caller then keeps the token path, which pins no pools).
/// \param avail_bytes  available physical RAM at planning time (ullAvailPhys / MemAvailable)
inline size_t ram_prompt_tokens_fit(size_t phys_bytes, size_t headroom_bytes, bool auto_headroom, size_t avail_bytes,
                                    size_t fixed_bytes, size_t per_token_bytes, size_t mapped_bytes, size_t requested) {
    if (requested < 2) return requested;
    const size_t margin = ram_keep_bytes(phys_bytes, headroom_bytes, auto_headroom, 0, 0);
    const size_t taken = fixed_bytes + mapped_bytes;
    if (avail_bytes <= margin + taken) return 0;
    if (per_token_bytes == 0) return requested;
    const size_t fit = (avail_bytes - margin - taken) / per_token_bytes;
    if (fit < 2) return 0;
    return std::min(requested, fit);
}

/// The pinned expert tier's ceiling: 108 GiB, the RAM share the GPTQ-v2 pack was measured at (its 114 GiB less the VRAM
/// slots).  A bigger box pins no more than the pack keeps in RAM.
constexpr size_t kRamTierCeilingBytes = (size_t)108 << 30;
/// The RAM the automatic tier always leaves free: the 10 GB watchdog floor (10 GiB here, a little more than 10 GB).
constexpr size_t kRamFloorBytes = (size_t)10 << 30;

/// The automatic pinned-tier budget, sized from the RAM the box has at planning time: the smallest of the room physical
/// RAM leaves, the available RAM less what the box keeps free (ram_keep_bytes, and at least the 10 GiB floor), and the
/// 108 GiB ceiling.  A smaller box therefore picks a smaller tier, and no box goes below the floor.
/// \param room_bytes   physical RAM less the headroom and the overhead (the engine's room)
/// \param avail_bytes  available RAM at planning time
/// \param keep_bytes   ram_keep_bytes() for this box
inline size_t ram_auto_tier_bytes(size_t room_bytes, size_t avail_bytes, size_t keep_bytes) {
    const size_t keep = std::max(keep_bytes, kRamFloorBytes);
    const size_t free_now = avail_bytes > keep ? avail_bytes - keep : 0;
    return std::min(std::min(room_bytes, free_now), kRamTierCeilingBytes);
}

/// The budget to hand the slot planners so an automatic tier reserves its real bytes.  The planners count a layer's slots as
/// budget / (MoE layers x the largest expert).  Layers whose experts are smaller than the largest would otherwise leave
/// part of the tier unused.  With this budget the count is tier / (the sum of one expert per MoE layer), so the reserved
/// bytes reach the tier and never pass it.
/// \param tier_bytes        ram_auto_tier_bytes()
/// \param expert_bytes_sum  one expert's bytes, summed over the MoE layers
/// \param moe_layers        MoE layers
/// \param max_expert_bytes  the largest expert
inline size_t ram_slot_budget_bytes(size_t tier_bytes, size_t expert_bytes_sum, size_t moe_layers, size_t max_expert_bytes) {
    if (expert_bytes_sum == 0) return tier_bytes;
    const size_t slots_each = tier_bytes / expert_bytes_sum;
    return (slots_each + 1) * moe_layers * max_expert_bytes - 1;
}

/// The pinned tier's per-layer slot counts under a routing profile: the frequency walk, as the engine runs it.
///
/// The walk hands out slots hottest first.  Each profile entry offers its layer one more slot for that entry's expert,
/// unless that expert already has one of the layer's VRAM slots, and a slot costs its own layer's expert bytes.  Whatever
/// the profile does not cover (a truncated profile, or a layer with no entry) is then filled round-robin from the budget
/// that is left, so the tier is not left half empty.
///
/// The walk never spends more than the uniform plan's bytes: the same count on every MoE layer, the floor of
/// budget / (MoE layers x the largest expert) capped at `cap`, each layer paying its own expert bytes.  The walk used to
/// fill the whole budget, which left less free RAM than the uniform plan and tripped the watchdog floor.
///
/// \param profile        (MoE layer, expert) pairs, hottest first; the MoE layer counts from `leading_dense`
/// \param leading_dense  the index in `moe` of the first MoE layer
/// \param moe            per layer: is it an MoE layer
/// \param vram_n         per layer: its VRAM slots (the first profile entries of a layer are those)
/// \param expert_bytes   per layer: the bytes of one expert
/// \param cap            experts per layer (n_expert)
/// \param budget_bytes   the pinned tier's byte budget
/// \return the RAM slots of each layer (0 for a layer that is not an MoE layer)
inline std::vector<int> ram_slots_by_profile(const std::vector<std::pair<int32_t, int32_t>>& profile, int leading_dense,
                                             const std::vector<char>& moe, const std::vector<int>& vram_n,
                                             const std::vector<size_t>& expert_bytes, int cap, size_t budget_bytes) {
    const size_t n = moe.size();
    std::vector<int> ram_n(n, 0);
    size_t moe_layers = 0, max_bytes = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!moe[i]) continue;
        ++moe_layers;
        max_bytes = std::max(max_bytes, expert_bytes[i]);
    }
    if (moe_layers == 0 || max_bytes == 0 || cap <= 0) return ram_n;

    // The uniform plan, sized as the engine sizes it, and the bytes it reserves.
    const size_t uniform = std::min<size_t>(budget_bytes / (moe_layers * max_bytes), (size_t)cap);
    size_t uniform_bytes = 0;
    for (size_t i = 0; i < n; ++i)
        if (moe[i]) uniform_bytes += uniform * expert_bytes[i];
    const size_t limit = std::min(budget_bytes, uniform_bytes);

    std::vector<int> seen(n, 0);
    size_t spent = 0;
    for (const auto& entry : profile) {
        const int32_t layer = entry.first;
        if (layer < 0 || (size_t)layer + (size_t)leading_dense >= n) continue;
        const size_t i = (size_t)layer + (size_t)leading_dense;
        if (!moe[i]) continue;
        const int s = seen[i]++;
        if (s < vram_n[i]) continue;   // the first entries of this layer are its VRAM slots
        if (ram_n[i] >= cap) continue;
        if (spent + expert_bytes[i] > limit) break;
        ++ram_n[i];
        spent += expert_bytes[i];
    }
    for (bool progress = true; progress;) {
        progress = false;
        for (size_t i = 0; i < n; ++i) {
            if (!moe[i] || ram_n[i] >= cap) continue;
            if (spent + expert_bytes[i] > limit) continue;
            ++ram_n[i];
            spent += expert_bytes[i];
            progress = true;
        }
    }
    return ram_n;
}

} // namespace strata::minimax

#endif // STRATA_MINIMAX_RAM_BUDGET_HPP
