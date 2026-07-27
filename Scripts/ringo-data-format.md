RinGoData v2

All multibyte values are little-endian. There is one file per shard. v2 is NOT backward
compatible with v1 (dropped the one-hot policy index and added per-sample validity flags for
distillation support, see WP-2 in `docs/strategy/TRAINING_FIX_PLAN.md`); a v1 shard must be regenerated with
the current `ringo makedata` (~23 minutes for the full corpus) -- `TrainingShardReader` and
`RinGoShard.read` both reject any other version with an explicit error rather than silently
misreading the bytes.

Header, in order:
- 4 bytes: ASCII magic `NNGD`
- u32: version, always 2
- u32: nnLen
- u32: numSpatial, always 22
- u32: numGlobal, always 19
- u32: sampleCount

Each sample, in order:
- f32[22*nnLen*nnLen]: spatial features in NHWC order from fillRowV7, from the side-to-move perspective
- f32[19]: global features
- f32[nnLen*nnLen+1]: policyTarget, a dense probability distribution over all moves;
0...nnLen*nnLen-1 is y*nnLen+x, and nnLen*nnLen is pass. Teacherless `makedata` writes a
one-hot vector (1.0 at the played move); a teacher-distilled shard may write a soft
distribution instead.
- f32[3]: valueTarget in win, loss, noResult order, from the side-to-move perspective. May be a
soft probability (e.g. a teacher's win-rate estimate) rather than a one-hot outcome.
- f32: scoreTarget, final score difference including komi, from the side-to-move perspective
- u8: scoreValid; 1 if scoreTarget reflects a genuine scored/estimated result, 0 if it is a
synthetic placeholder (e.g. the resignation proxy) that consumers must exclude from the score
loss and its normalization
- i8[nnLen*nnLen]: ownershipTarget; +1 is side-to-move owned, -1 is opponent owned, and 0 is
neutral/dame/seki-shared, using final-position area classification
- u8: ownershipValid; 1 if ownershipTarget reflects a genuine final-position label, 0 if it
must be excluded from the ownership loss and its normalization (e.g. a resignation, whose final
board position is not a reliable ownership label -- docs/reviews/TRAINING_PROCESS_AUDIT.md P0-3)

Jigo games are skipped because there is no draw value-target component. For resignations, the
score target is a deliberately synthetic, clamped proxy: +15.0 for the winner and -15.0 for
the loser, and BOTH `scoreValid` and `ownershipValid` are false.

## v1 -> v2 byte-layout deltas

- Header: `version` is now `2` (was `1`); everything else in the header is unchanged.
- `policyTarget` changed from `u16` (a single class index, 2 bytes) to `f32[nnLen*nnLen+1]` (a
  dense distribution, `(nnLen*nnLen+1)*4` bytes) -- e.g. for 9x9 (nnLen=9, area=81), 2 bytes ->
  328 bytes.
- `valueTarget` is unchanged in layout (`f32[3]`) but is no longer guaranteed one-hot; a
  teacher-distilled shard may write soft win/loss/noResult probabilities.
- A new `scoreValid` `u8` is inserted immediately after `scoreTarget` (before `ownershipTarget`).
- A new `ownershipValid` `u8` is appended immediately after `ownershipTarget` (the new last field
  in the sample).
- Net effect on `RinGoDataFormat.sampleSize(nnLen:)` for nnLen=9 (area=81): v1 was
  `(22*81 + 19 + 3 + 1)*4 + 2 + 81 = 7,303` bytes/sample; v2 is
  `(22*81 + 19 + 82 + 3 + 1)*4 + 1 + 81 + 1 = 7,631` bytes/sample (+328 bytes/sample, i.e.
  +326 from widening the policy field, +2 from the two new validity bytes).
