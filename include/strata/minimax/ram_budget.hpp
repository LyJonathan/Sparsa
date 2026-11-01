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
// Header-only, so the arithmetic can be unit-tested without a GPU (tests/minimax/ram_budget_test.cpp).
#ifndef STRATA_MINIMAX_RAM_BUDGET_HPP
#define STRATA_MINIMAX_RAM_BUDGET_HPP

#include <algorithm>
#include <cstddef>

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

} // namespace strata::minimax

#endif // STRATA_MINIMAX_RAM_BUDGET_HPP
