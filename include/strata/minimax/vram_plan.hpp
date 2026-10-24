// include/strata/minimax/vram_plan.hpp - the automatic VRAM-slot count as a pure function.
//
// The engine sizes the expert slots from the free VRAM it sees after the dense weights, the KV and the token
// workspace.  The batched-prefill workspace is allocated later (init_prefill), so the old sizing never subtracted it
// and a card large enough for the slots to fill the card died in cudaMalloc at startup.  The rule here is the fixed
// one: reserve the prefill workspace first, then give the rest to slots.  It is header-only so the arithmetic can be
// unit-tested without a GPU.
#ifndef STRATA_MINIMAX_VRAM_PLAN_HPP
#define STRATA_MINIMAX_VRAM_PLAN_HPP

#include <cstddef>
#include <cstdint>

namespace strata::minimax {

struct VramSlotPlan {
    size_t usable_bytes = 0;    ///< free - reserve - prefill, the bytes the slots may take
    size_t slot_bytes = 0;      ///< slots * moe_layers * expert_bytes on this device
    int slots = 0;              ///< per-layer slot count
    bool prefill_reserved = false;   ///< the plan subtracted a batched-prefill workspace
};

/// \param free_bytes     cudaMemGetInfo().free at sizing time (dense weights, KV and token workspace already taken)
/// \param reserve_bytes  --vram-reserve-mib, kept for the driver/display
/// \param prefill_bytes  the batched-prefill workspace init_prefill will allocate on this device (0 when off)
/// \param expert_bytes   one expert's bytes (the largest on this device)
/// \param moe_layers     MoE layers on this device
/// \param n_expert       experts per layer (the maximum a slot count can hold)
/// \param requested      explicit <c>--minimax-vram-slots-per-layer</c>; -1 means auto
///
/// Auto keeps the invariant <c>reserve + prefill + slot_bytes &lt;= free_bytes</c>.  It does not clamp the slot count
/// to leave the prefill space unclaimed only when it is needed: the prefill allocation is not optional once
/// <c>--minimax-prefill-batch</c> is on, so it is always subtracted.
inline VramSlotPlan plan_vram_slots(size_t free_bytes, size_t reserve_bytes, size_t prefill_bytes,
                                    size_t expert_bytes, int moe_layers, int n_expert, int requested) {
    VramSlotPlan p;
    const size_t after_reserve = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
    const size_t per_layer =
        expert_bytes * (size_t)(moe_layers > 0 ? moe_layers : 0);
    if (requested >= 0) {
        int q = requested;
        if (q > n_expert) q = n_expert;
        if (q < 0) q = 0;
        p.usable_bytes = after_reserve;
        p.slots = q;
        p.slot_bytes = per_layer * (size_t)q;
        return p;
    }
    const size_t prefill = prefill_bytes < after_reserve ? prefill_bytes : after_reserve;
    const size_t budget = after_reserve - prefill;
    int q = per_layer > 0 ? (int)(budget / per_layer) : 0;
    if (q > n_expert) q = n_expert;
    if (q < 0) q = 0;
    p.usable_bytes = budget;
    p.slots = q;
    p.slot_bytes = per_layer * (size_t)q;
    p.prefill_reserved = prefill > 0;
    return p;
}

}  // namespace strata::minimax

#endif  // STRATA_MINIMAX_VRAM_PLAN_HPP
