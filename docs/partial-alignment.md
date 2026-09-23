# Partial alignment

`partial` is an experimental alignment mode for material with short matching fragments, inserted scenes, and ambiguous repeated content. It is opt-in; `auto` continues to use the established alignment pipeline.

```bash
adsync analyze video.mkv description.m4a --mode partial --report alignment.json
adsync sync video.mkv description.m4a --mode partial --report alignment.json
adsync season "Show videos" "AD tracks" --mode partial --output-dir "Finished"
```

Start with `analyze` to inspect the map. `sync` can render a provisional result when some fragments match. Unplaced parts of the target timeline contain silence in the added AD stream, and unplaced source audio is omitted from that stream. The source files remain intact. The report lists these intervals as **unmeasured**: lack of matching evidence does not establish that a scene is absent.

No supported map means no output media. Unresolved gaps or competing timing paths set `alignment_review_required`, independently of the numerical confidence threshold. Season processing retains provisional results in staging for review and applies its usual rendered-audio and video-preservation checks before publication.

## How it works

The existing multiband correlation front end retrieves several candidate positions for each two-second source window, sampled every second. Fingerprints guide the search location. Windows without correlation candidates can use raw landmark matches, provided those matches occupy distinct times spanning both halves of that same window. This fallback keeps competing offsets and records fingerprint provenance. It does not insert candidates into an already measured window or borrow support from neighbouring content.

The decoder treats candidates as nodes in a sparse, ordered graph. A path can continue through matching content, cross an edit, or leave an interval unmeasured. Gap-opening and extension costs discourage many isolated omissions while allowing a coherent gap. Transitions must move forward on both timelines. Within a matching run, the time-scale change is bounded by `max_stretch`.

Forward and backward dynamic programs compute the best complete path through each admitted candidate. The difference between a selected path and a competing location is a **score margin**, not a calibrated probability. Two strong but indistinguishable repeated scenes should produce ambiguity instead of a high-certainty verdict based only on correlation amplitude.

The fitter uses supported runs directly, including short runs with at least three consistent measurements. Piecewise-linear maps preserve measured coordinates while bounding local rates. It does not pass them through the older minimum-duration filtering or interpolate a curve across unsupported material. Reconstruction uses ADSync's existing warp renderer and its pitch-preserving treatment of time-scale changes.

## Reading the report

`timing_debug.partial_alignment` records:

- `matched_intervals`: measured source-to-target spans.
- `source_gaps` and `target_gaps`: unmeasured intervals on either timeline.
- `ambiguous_ranges`: source regions with competing plausible paths.
- `windows`: selected observations and alternative-path score margins.
- `candidates` and `path_score`: search size and the selected graph objective.

Those measurements describe the alignment before a manual playback nudge. `playback_offset_adjust_sec` records that nudge; the standard `fitted_pchip`, `segment_ranges`, and reported anchors describe the adjusted map used for rendering. Window support is a finite measurement interval, not a claim of a sample-exact edit boundary.

Fingerprint residuals are consistency measurements; they are not independent validation when fingerprints contributed to the fit. The season workflow separately checks rendered audio.

## Research behind the design

[Hidden State Time Warping](https://www.mdpi.com/2076-3417/12/8/3783) explicitly separates matching and nonmatching regions. [Drop-DTW](https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html) incorporates dropping outliers into sequence alignment. [Affine-gap alignment](https://pubmed.ncbi.nlm.nih.gov/7166760/) distinguishes the cost of starting a gap from extending it. These ideas motivate the representation here; this implementation is a sparse candidate-graph decoder, not a reproduction of those papers or a differentiable DTW implementation.

Other research directions have different roles. [Memory-restricted multiscale DTW](https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2016_PraetzlichDriedgerMueller_MrMsDTW_ICASSP.pdf) addresses long-sequence resource use. [Frequency-sliding GCC](https://arxiv.org/abs/1910.08838) combines sub-band delay evidence under interference. [Neural audio fingerprints](https://arxiv.org/abs/2010.11910) could supply alternative candidates when ordinary landmark retrieval fails. None of those methods is implemented by this mode.

## Limits and evaluation

This mode still depends on candidate retrieval. If the correct scene never enters the candidate set, its score margin cannot reveal that missing alternative. A monotone path also cannot model genuinely reordered scenes. Narration is overlapping interference, not automatically an inserted time interval; different mixes and narration-heavy passages can leave insufficient evidence.

The decoder has quadratic time cost in the number of candidate nodes and linear state memory. It admits up to 40,000 eligible candidates, then stops explicitly rather than silently discarding hypotheses. For long inputs, reduce `--warp-candidates` on `sync` or use the established modes. Partial mode currently fixes the analysis step at one second. The default search radius remains bounded at 240 seconds; `--warp-search-radius` can override it on `sync`.

Run the reproducible adversarial harness outside the repository:

```bash
python tools/partial_alignment_harness.py --workdir "alignment-evaluation" --modes warp partial
```

It reports timing error and placement coverage separately for each known fragment, along with false placement inside known unmatched source material. A short missed fragment must remain visible even when most of a track is correct. Generated examples do not establish superiority over another aligner or replace held-out real-media evaluation.

The initial regression corpus used seed `20260923`, 0.1-second probes, a one-second exclusion around known edit boundaries, and a 50 ms placement tolerance. Results were the same with fingerprints enabled and disabled:

| Analysis-map check | Existing warp route | Partial |
|---|---:|---:|
| Clean matched interior correctly placed | 100% | 100% |
| Six-second matched island correctly placed | 0% | 100% |
| Five-second matched island correctly placed | 0% | 100% |
| False placements inside known inserted source material | 100% | 0% |
| Repeated scene identified as ambiguous | 0% | 100% |
| Unrelated source assigned target positions | 100% | 0% |

These are coordinate-map measurements, including maps from legacy fallback modes. They do not imply that the season workflow would publish a rejected result. A separate generated-container test checks the encoded AD around the short island, silence inside the target gap, and unchanged compressed video.

There is a coverage trade-off. On a separate 120-second real AD excerpt with a nominal identity map, the existing warp route placed 100% of sampled interior points within 50 ms; partial placed 70.3% and left the rest unmeasured. Neither placed a sampled point outside that tolerance. The p95 errors among placed points were 17.7 ms and 22.5 ms respectively. Partial correctly required review. That excerpt is not distributed with the repository, and one excerpt is not a broad real-world accuracy benchmark.
