# Season processing

The season command processes one show's video and AD folders, including multiple seasons. It follows the same alignment and publication path as a single-file sync.

## Matching

Explicit episode identifiers take priority. Season folders provide context for filenames with only episode numbers. Numeric-sequence inference requires evidence in both input folders and is disclosed in the preview. Duplicate candidates, conflicting identifiers and multi-episode files remain unresolved. File order is never a pairing rule.

`--dry-run` previews matches without opening media or creating outputs. Normal execution probes media before scheduling. `--strict` stops on unresolved inputs; otherwise the command processes clear matches and returns a review status for the rest.

## Processing

Each episode runs in a separate process with its own log and JSON report. Preparation selects the requested original language and downmixes multichannel audio in the final mux, avoiding a full intermediate video. Already suitable original audio is copied. `--no-prep` preserves the original stream set.

The parent limits concurrent jobs using usable CPU count, available RAM, measured GPU availability and estimated episode memory. Each worker receives a CPU thread budget for numerical libraries, correlation workers and FFmpeg. This avoids multiplying each library's automatic thread pool by the number of episodes. The estimates reserve memory for other applications; `--jobs` and `--threads` provide explicit control within the detected resource budget.

## Resume and failures

Checkpoint writes are atomic. A saved completion is reusable only when input content, processing settings, algorithm and quality-policy revisions, and the hashes of the output, report and verification record still match. Changing a performance setting does not invalidate completed media. Existing outputs without a matching receipt require `--overwrite`.

Each episode is rendered locally and checked before publication. Checks compare shared audio around edits, sparse matching regions and the ending, and verify that the original compressed video is preserved. Low-confidence or inconclusive results remain in local staging with their evidence. A failed episode does not stop independent jobs. Cancellation stops worker processes and their FFmpeg children, then preserves the checkpoint. Completed output remains recoverable if publication to a shared destination fails.

Publication is serialized by destination filesystem on one computer. A cross-filesystem copy is hashed before it becomes visible under its final name. The final commit record binds media, report and quality results; an interrupted bundle without that record is not considered completed. Progress records distinguish processed, locally verified, published and transfer-verified states. State and output reservations do not coordinate writers on separate computers.

Preprocessed audio and basic features are cached using source SHA256, selected audio stream, analysis settings and an explicit analysis revision. `ADSYNC_CACHE_DIR` chooses the cache directory; `off` disables it. The default is the user's local application cache. Corrupt cache entries are rebuilt. `ADSYNC_WORK_DIR` chooses the local episode staging directory. Keep these directories on fast local storage when the final library is on a slower drive or network share.

Content identity requires matching soundtrack evidence distributed through the episode. A matching opening theme or high raw match count is insufficient. These checks establish shared content and timing; they do not verify the meaning of every narration sentence.

An abrupt crash can leave a reservation file. The lock error identifies its owner and path; confirm that process has stopped before removing a stale reservation. Normal completion and handled interruption release their reservations automatically.

## Validation

Matching tests cover explicit identifiers, directory context, numeric sequences, missing files, duplicates and generated-output exclusions. Runner tests cover resume, changed inputs, output conflicts, interrupted jobs and bounded concurrency. Hardware tests cover CPU-only and CUDA systems with different memory budgets. Throughput measurements compare the same generated episode pack with several worker counts; speed claims report total elapsed time and the tested workload.
