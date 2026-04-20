# ADSync

**Fan-made audio description tracks, perfectly aligned with your video — in one command.**

Accessibility should not depend on whether someone happened to rip the right cut. ADSync takes any audio description (AD) track and locks it onto your video file, automatically, even when the two were made from different edits.

---

## Why this exists

Most streaming services do not ship audio description. Blind and visually impaired viewers rely on community-made AD tracks to actually enjoy what they watch.

The problem: a fan's AD track was recorded against *their* copy of the episode. Yours is almost never the same cut — different intros, different ad breaks, different frame rates, different anything. Drop the AD on top and by minute ten it is narrating the wrong scene.

ADSync fixes that. You hand it a video file and an unsynced AD track. It gives you back a single MKV with the AD perfectly slotted in as a selectable audio stream. No manual nudging, no Audacity, no spreadsheet of timestamps.

## Who it is for

- **Blind and visually impaired viewers** — or anyone building tools for them
- **Accessibility curators** batch-aligning libraries of community AD tracks
- **Archivists** pairing old descriptive-audio recordings with modern releases
- **Media tinkerers** who want one command instead of an evening of nudging

## What makes this different

Most existing AD-sync tools chop the description into fixed chunks and align each one independently. That works for simple drift, but it falls apart when the cuts don't match: you get audio that skips back and replays the same second, jumps forward, then rewinds when the next chunk disagrees with the last. Stutter loops, phantom repeats, mid-sentence rewinds. Unlistenable for the person who actually needs the narration.

ADSync refuses to make per-chunk decisions in the first place. Instead:

- It builds a **top-K candidate lattice** of plausible offsets across every analysis window
- Runs a **Viterbi / DP decoder** with explicit penalties on offset jumps and curvature — so the track is solved **globally**, not locally
- Fits a **shape-preserving monotone PCHIP warp** through the decoded points — guaranteed never to run backwards in time
- Renders the final audio from a **continuous time-map**, sample by sample — no seams, no re-glueing of chunks

The outcome: no stutter loops, no backwards audio, no phantom repeats. Either the alignment holds cleanly or the confidence score flags it up front — ADSync won't silently produce a broken mix and hope you don't notice.

## What you get

- **One-shot sync → mux.** Input two files, output one MKV with the AD embedded as a tagged, selectable track.
- **Handles the ugly cases.** Constant offset, linear clock drift, *and* discontinuous edits (different ad-break placement, missing scenes, inserted recaps) — all in one pipeline.
- **Globally-optimised warp alignment.** A Viterbi decoder walks a candidate lattice and fits a shape-preserving monotone warp — so the whole track stays consistent instead of each chunk guessing independently.
- **Sub-sample accuracy.** Parabolic interpolation around cross-correlation peaks gives ~1–3 ms precision.
- **Fast and honest.** Streams PCM straight into FFmpeg — no huge temp files. Every run produces a confidence score and a warnings list so you know when to trust the output and when to review.
- **Debug mode that actually helps.** Dumps feature CSVs, plots, and intermediate WAVs when you want to understand what the aligner saw.

## How it works

ADSync tries four alignment strategies in order of complexity and uses whichever earns a high enough confidence score:

| Mode | When it wins | What it does |
|---|---|---|
| `offset` | The AD is a clean shift of the video (same cut, different start time) | Single global offset via normalised cross-correlation on downsampled raw audio |
| `drift` | Both tracks match but sample rates or clocks differ slightly | Measures offset at several points, fits a weighted linear drift model |
| `warp` *(default fallback)* | Cuts differ — inserted scenes, missing recaps, shifted ad breaks | Builds a top-K candidate lattice per window, runs a DP decoder with jump/curvature penalties + speech-rich bonuses, then fits a shape-preserving PCHIP warp and renders the output from a continuous time-map |
| `piecewise` *(legacy)* | Present for comparison with the old stitching approach | Anchor search + piecewise map with crossfades |

Warp mode is the default when offset/drift aren't enough. It replaces the older piecewise stitcher because a global DP decoder refuses to chase musical false-positives the way per-chunk decisions do.

## Installation

Requirements:

- Python 3.10+
- FFmpeg and ffprobe on your `PATH` ([ffmpeg.org](https://ffmpeg.org))

Then:

```bash
git clone https://github.com/YOUR-USERNAME/ADSync.git
cd ADSync
pip install -e .
```

For development / running tests:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
adsync sync episode.mkv ad_track.m4a -o episode.with-ad.mkv
```

That's it. Open the resulting MKV in VLC, MPV, Plex, Jellyfin — pick the "Audio Description" track and press play.

## Commands

```bash
# Full sync — produces the final MKV with AD embedded
adsync sync episode.mkv ad_track.m4a -o episode.synced.mkv

# Analysis only — writes a JSON report, no output file
adsync analyze episode.mkv ad_track.m4a --report report.json

# Debug — dumps feature CSVs, anchor plots, and intermediate WAVs
adsync debug episode.mkv ad_track.m4a --workdir debug_out

# Mux only — you already have a synced AD file, just drop it in the container
adsync mux episode.mkv already_synced_ad.m4a -o final.mkv
```

## Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `auto` | `auto`, `offset`, `drift`, `warp`, or `piecewise` |
| `--codec` | `libopus` | Encoder for the AD track (Opus is tiny and clean for speech) |
| `--bitrate` | `96k` | AD track bitrate |
| `--language` | `eng` | Language tag written into the MKV metadata |
| `--ad-title` | `Audio Description` | Title tag for the track |
| `--offset-adjust` | `0.0` | Manual nudge in seconds (positive = push AD later). Handy when sync is great but you want the narration to land slightly before/after |
| `--confidence-threshold` | `0.70` | Exit code 1 below this, so batch scripts can auto-flag risky runs |
| `--warp-lambda-jump` | `2.0` | Warp DP penalty for offset jumps |
| `--warp-lambda-curve` | `5.0` | Warp DP penalty for curvature |
| `--warp-lambda-speech` | `0.3` | Warp bonus for speech-rich windows |
| `--warp-candidates` | `5` | Max offset candidates per analysis window |

Run `adsync <command> --help` for the full list.

## Tested on *From*

ADSync has been used end-to-end on episodes of **From** (the MGM+ horror/mystery series), pairing fan-contributed AD tracks with retail releases. Across the episodes tested so far, sync has held up well — including a case where the AD and video had a genuine edit discontinuity mid-episode, which warp mode correctly handled as a single offset jump rather than trying to force a linear drift fit.

If you have the tracks, it works.

## Known artifacts

- **Very brief pitch drift (~a few seconds, self-correcting).** Occasionally you can hear the narration's pitch shift slightly before snapping back. This is a side effect of time-warping the AD to match the video timeline. On the roadmap to fix shortly — likely by swapping the inner resampler for a phase-vocoder / WSOLA path so stretch doesn't touch pitch.

## Roadmap

- [ ] Eliminate the transient pitch drift during warped stretching (WSOLA or phase-vocoder resampler)
- [ ] Optional GPU-accelerated cross-correlation for faster runs on long files
- [ ] Precomputed AD offset database / cache
- [ ] Web UI for non-technical users
- [ ] Automatic detection of matching AD tracks from a library

## Contributing

Issues and PRs welcome, especially:

- Bug reports with a JSON report attached (from `adsync analyze`)
- Hard cases — tracks that come out wrong (even if you can only describe the source material, not share it)
- Improvements to the warp decoder penalties, confidence model, or rendering

Please keep PRs focused. One change per PR makes review fast.

## License

MIT. See [LICENSE](LICENSE).

## A note on content

ADSync does not ship, download, or distribute any copyrighted video or audio. It is a local tool that operates on files you already have. Audio description tracks were created by dedicated volunteers — please credit them where you can.
