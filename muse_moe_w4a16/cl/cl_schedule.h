// Per-call expert-slice schedule of the MoE cluster kernel (docs/plans/2026-11-04-moe-cluster-rewrite.md 3.1).
// Written once per call by the prologue (cl_prologue.cu, or the route kernel in the layer), read by every cluster.
// cl_schedule.py builds the same bytes on the host; a layout change must be made in both files.
//
// Byte layout (little-endian, every field 4 bytes):
//   offset  size  field
//        0     4  n_slices           S x distinct local experts; -1 means overflow (see below)
//        4   256  slice_expert[64]   local expert id (0 .. count-1) of slice i
//      260   256  slice_col0[64]     first intermediate column of slice i, multiple of 256
//      516   256  slice_ncol[64]     columns in slice i, multiple of 256
//      772   256  slice_tok0[64]     first tok_list entry of slice i's expert
//     1028   256  slice_ntok[64]     (token, slot) pairs routed to slice i's expert
//     1284  1024  tok_list[256]      token ids grouped by ascending local expert, ascending (token, slot) inside one
//     2308  1024  tok_gate[256]      routing weight of the matching tok_list entry, copied bit for bit
//     3332   128  expert_nslices[32] slices of each local expert, 0 if unrouted
//     3460     4  next_slice         dynamic claim counter, initialised to the number of clusters launched
//   sizeof = 3464 bytes = 866 words.
//
// Slices are ordered by ascending local expert, then ascending col0. Every entry past n_slices (or past the pair
// total in tok_list / tok_gate) is zero, so two writers given the same inputs produce identical bytes.
// Overflow: more than CL_MAX_PAIRS local pairs (reachable at T > 32) or more than CL_MAX_SLICES slices. The struct
// is then all zero except n_slices = -1 and next_slice, and the caller must take the fallback MoE path.
#pragma once

#define CL_MAX_SLICES 64
#define CL_MAX_EXPERTS 32
#define CL_MAX_PAIRS (CL_MAX_EXPERTS * 8)
#define CL_COL_CHUNK 256
#define CL_N_CHUNKS 16
#define CL_SCHED_WORDS 866

#define CL_POLICY_S1 0
#define CL_POLICY_S2 1
#define CL_POLICY_FILL 2

struct ClSchedule {
  int n_slices;
  int slice_expert[CL_MAX_SLICES];
  int slice_col0[CL_MAX_SLICES];
  int slice_ncol[CL_MAX_SLICES];
  int slice_tok0[CL_MAX_SLICES];
  int slice_ntok[CL_MAX_SLICES];
  int tok_list[CL_MAX_PAIRS];
  float tok_gate[CL_MAX_PAIRS];
  int expert_nslices[CL_MAX_EXPERTS];
  int next_slice;
};

static_assert(sizeof(ClSchedule) == 3464, "ClSchedule layout changed; update cl_schedule.py");
static_assert(sizeof(ClSchedule) == CL_SCHED_WORDS * 4, "CL_SCHED_WORDS");
static_assert(__builtin_offsetof(ClSchedule, slice_expert) == 4, "offset");
static_assert(__builtin_offsetof(ClSchedule, slice_col0) == 260, "offset");
static_assert(__builtin_offsetof(ClSchedule, slice_ncol) == 516, "offset");
static_assert(__builtin_offsetof(ClSchedule, slice_tok0) == 772, "offset");
static_assert(__builtin_offsetof(ClSchedule, slice_ntok) == 1028, "offset");
static_assert(__builtin_offsetof(ClSchedule, tok_list) == 1284, "offset");
static_assert(__builtin_offsetof(ClSchedule, tok_gate) == 2308, "offset");
static_assert(__builtin_offsetof(ClSchedule, expert_nslices) == 3332, "offset");
static_assert(__builtin_offsetof(ClSchedule, next_slice) == 3460, "offset");
