# Packed calibration and durable interval correction

## Fixed-point recognition

The input is six signed Q8 channels, with 256 integer units per g of acceleration and per rad/s of angular velocity. Each 100-sample window spans two seconds at 50 Hz. The C encoder produces the mean, mean absolute deviation around the rounded mean, mean absolute adjacent difference, and mean absolute magnitude for each channel. The result is a 24-dimensional vector. Source-subject means and inverse-scale multipliers are fitted on the host. The fixed-point projection clips each coordinate to [−2047, 2047]. Each class centroid includes eight pseudo-observations at the source centroid. Inference uses minimum squared Euclidean distance with a lowest-index tie break.

Four windows with a 25-sample stride form a one-second editing block by their anchor times. Their receptive fields extend beyond that second. All evaluation windows lie wholly in a labeled source segment. Selected excerpts are ordered by acquisition. Excerpts from different recordings use consecutive ledger indices, which are separate from wall-clock timestamps.

## Exact representation and edits

For a block with vectors z₁ to zₙ, store their coordinate sums and the count n. Every coordinate sum lies in [−8188, 8188] because n ≤ 4. Signed 14-bit two's-complement represents [−8192, 8191], so 24 coordinates occupy 42 bytes. The count and original class add two bytes. A group of four consecutive coordinates occupies seven bytes in little-endian bit order. The device stages the 44-byte record after extracting features from the raw input windows. Class aggregates use signed 32-bit coordinates, which cover the maximum magnitude of 128 × 4 × 2047 = 1,048,064.

A half-open edit interval moves all blocks currently labeled A to B. For each block, subtract its sum and count from A and add them to B. Initially the aggregates equal a rebuild from the enrolled vectors. Each edit preserves this equality. The same deterministic rounding and tie rule therefore produce identical centroids and predictions. The independent Python oracle rebuilds from the unaggregated vectors. The smallest editable increment is one block. Multiple source classes use separate transactions.

## Durable state and request semantics

The immutable calibration base is provisioned under an epoch. The journal writes six 32-bit words for each edit. The first five contain the request ID, half-open range, source and destination classes, epoch, and CRC-32. The sixth word commits the transaction. After programming the commit marker, firmware updates volatile class statistics and labels. The reader replays structurally valid committed records in ID order. A partly written slot stays unavailable until compaction.

Two flash banks alternate checkpoint generations. Compaction erases the inactive bank, then writes the current labels, high-water ID, epoch, generation, latest valid request record, CRC, complemented population count, and final commit word. The previous valid bank survives until a later compaction. The checkpoint contains 45 words in this implementation. A restore checks the epoch and integrity fields and chooses the newest valid generation. Boot reconstructs class sums from the immutable base under the checkpoint labels, then replays the active journal.

An identical retry for the latest committed ID receives a duplicate success acknowledgement without any flash write, including after compaction and reset. A different payload for that ID returns a conflict. Other IDs retained in the active journal are also checked. Compaction discards the older payloads, and requests for these IDs return a stale status. Callers serialize mutations and use one strictly increasing request stream. When a journal is full, the caller checkpoints and retries the new request.

Under the assumed NOR model, a program cut can leave a subset of the intended 1→0 transitions. A partly written commit word becomes valid only if every required zero bit is programmed, after the earlier words have completed. For an interrupted erase, any altered data or CRC word increases the number of one bits. The stored complemented count can only increase under 0→1 erasure, while the correct complement for the new data would decrease. The exact commit word separately detects erasure touching its bits. This argument assumes that erases affect only the inactive bank and that old and new epochs are provisioned through the explicit provisioning operation.

## Evaluation protocol

| Item | Setting |
|---|---|
| Data | HAPT has 30 participants. MotionSense has 24. Each uses six classes. |
| Source fitting | The target participant is excluded from normalization and prior fitting. |
| Target split | HAPT uses an early recording for calibration and later recording(s) for test. MotionSense uses trials 1–9 for calibration and 11–16 for test. |
| Calibration | Four consecutive blocks per class, 24 blocks and 96 vectors per participant and seed. |
| Seeds | 17, 43, and 101. |
| Label-error workload | Three simulated incorrectly labeled intervals per calibration run. Primary intervals last two blocks. One and three blocks are sensitivity settings. |
| Accuracy | Compute macro-F1 per participant and seed, average the seeds per participant, and resample participants for paired 95% bootstrap intervals. |
| Integer parity | Compare each method's labels, counts, sums, centroids, and predictions against full reconstruction. |
| Long streams | 162 streams of 64 committed edits on eight-slot journals, with repeated cuts, retries, compaction, and complete integer-state checks after each commit. |
| Cortex-M4 | Three participants per dataset, one per seed. The test harness builds the packed base from raw windows and checks expected state digests after reset. |

Simulated label errors follow a controlled annotation scenario, with known labels supplied as corrections. Dataset class and source-only results characterize the fixed-point model used here.

| Ablation | Change |
|---|---|
| No correction | Keep the incorrect labels. |
| Add-only update | Add evidence to the corrected class while leaving it in the original class. |
| No source prior | Set the source pseudo-count to zero. |
| Short history | Allow edits only in the last eight selected blocks. |
| Coarse edit grid | Require complete aligned pairs of one-second blocks. |
| Unpacked sums | Store the same integer sums in 32-bit coordinates. |
| No retry IDs | Assign a fresh request ID to a repeated submission. |
| No CRC validation | Bypass edit-record CRC verification on replay. |
| No checkpoint receipt | Retain the high-water ID while dropping the latest payload. |

## Baselines and resource accounting

- **Incremental vectors** retains all four int16 feature vectors per block and sums the edited blocks on demand. It shares the journal, CRC, compaction, and retry implementation.
- **RAM-cached vectors** retains the same vectors and reconstructs 100-byte block summaries in RAM at boot. The same journal then uses these cached summaries during edits. The table includes the extra cache bytes.
- **Full-vector rebuild** adds complete vector reconstruction after a durable edit. This reference illustrates the cost of replay-based correction.
- **Unpacked journal** uses 100-byte summaries in flash with the same journal and receipt semantics.
- **Two-bank snapshot** uses the same packed calibration base and latest-request receipt. It rewrites the label checkpoint in the inactive bank for each edit and applies the same commit and validation rules. Both protocols preserve the same latest-request acknowledgement under the tested resets.

The packed and vector journals each program 24 bytes per clean edit without erasing a sector. Compaction contributes one erase and 180 programmed bytes after 16 filled slots in the target configuration. Thus, a clean stream filling each journal amortizes to 35.25 programmed bytes and one-sixteenth of a sector erase per edit. The snapshot programs 180 bytes and erases one sector per clean edit. Interrupted writes and earlier compaction can increase the journal cost. Boot replay may favor the snapshot for long histories in other configurations.

The STM32F4 mapping reserves three 128 KiB sectors for isolation. This reservation is shared across implementations. Calibration bytes refer to the payload. Core instruction counts bracket command execution in Renode and exclude mailbox exchange and idle time. Static linked RAM excludes stack. The measurement mailbox holds test inputs and staging buffers. The host NOR model exercises specified program masks, partial erase, and cuts at boundaries. The Renode campaign uses Cortex-M SYSRESETREQ with modeled flash. The recovery guarantee is for already provisioned bases under the stated fault model. The CRC experiment flips a payload bit in the newest edit. Electrical brownouts, current, physical timing, wear, arbitrary corruption of older durable records, and resets during base provisioning are outside the experiment.
