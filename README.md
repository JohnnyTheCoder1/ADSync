# ADSync

Sync an audio description track to your copy of a video, and mux them into one MKV. One command.

## Why this exists

First time I tried this was with an episode of a Netflix show and an AD track I'd pulled from [audiovault.net](https://audiovault.net). I had VLC open on one screen and Windows Media Player on the other. Play the video, start the AD a few seconds behind, listen, pause, rewind, nudge the AD a second earlier, play again. By the time the first ad break hit, the two were off in a way no single offset was going to fix. I gave up around minute twelve.

Doing that for one episode is a slow evening. Doing it for a season is a problem. I'm blind, so this isn't really an accessibility project for me. It's just how I watch TV.

The reason it's hard: whoever recorded the AD did it against their copy of the episode, and yours is almost never the same cut. Different intro length, different ad breaks, different frame rate, an inserted recap, the occasional missing scene. Drop the AD on top of a mismatched cut and by minute ten it's narrating the wrong thing.

ADSync takes the video and the unsynced AD, does the alignment in one pass, and hands you back an MKV with the AD embedded as a selectable audio stream. If it can't find a confident alignment it tells you instead of quietly producing a broken mix.

## Who this is for

- Blind and visually impaired viewers, and people building tools for us.
- Accessibility curators batch-aligning libraries of AD tracks.
- Archivists pairing older descriptive-audio recordings with modern releases.
- Anyone who has spent an evening in Audacity nudging timestamps and decided once was enough.

## The approach

The older, simpler way to do this is to chop the AD into fixed chunks, align each chunk independently, and stitch the results back together with crossfades. That works well when the two sources are close, a constant offset or a gentle drift. ADSync keeps a mode like that around (`piecewise`) for the easy cases and for comparison.

It gets harder when the cuts don't match: different ad breaks, an inserted recap, a trimmed scene. Independent per-chunk decisions can disagree with their neighbours, and once they do, the stitched output can skip or double back on itself. Warp mode takes a different trade-off:

- Build a top-K candidate lattice of plausible offsets at every analysis window.
- Run a Viterbi / DP decoder with explicit penalties on offset jumps and curvature, so the whole track gets solved as one piece instead of window-by-window.
- Fit a shape-preserving monotone PCHIP warp through the decoded points, so the time-map never runs backwards.
- Render the final audio from that continuous time-map, sample by sample. No chunk seams to glue.

The report exposes uncertain regions, and season processing checks the rendered audio before publication.

## What you get

- One command in, one MKV out, with the AD embedded as a tagged, selectable track.
- Handles constant offset, linear clock drift, and discontinuous edits (ad-break changes, missing scenes, inserted recaps) in the same pipeline.
- Audio landmark fingerprinting (Shazam-style constellation hashes) maps which stretch of the AD sits at which offset **globally**, edits of any size are located before alignment runs, and AD regions that match nowhere in the video are called out as bad spots in the report.
- Automatic speed detection. A PAL-sourced AD runs 25/24 fast with the pitch up and looks like a total mismatch; ADSync probes the standard transfer ratios, measures the exact stretch, and corrects speed and pitch before aligning. The applied correction goes in the report.
- Repeated soundtrack content is filtered against monotone playback, so a credits song that also plays twenty minutes earlier can't drag a stretch of the AD to a physically impossible position.
- Global warp alignment: a Viterbi decoder walks the candidate lattice and fits a shape-preserving monotone warp, so the whole track is solved as one piece. When the two sources carry different mixes of the same material and correlation starves, the fingerprint matches themselves anchor the alignment.
- After fitting, the warp is checked back against the fingerprint matches and the residuals (median/p95 in ms) go in the report, "verified" is a measurement, not a mood.
- Sub-sample accuracy via parabolic interpolation around cross-correlation peaks, roughly 1–3 ms.
- Streams PCM straight into FFmpeg, no huge temp files. Every run produces a confidence score and a warnings list.
- Debug mode that dumps feature CSVs, plots, and intermediate WAVs when you want to see what the aligner saw.

## How it works

ADSync tries four alignment strategies in order of complexity, and uses whichever one clears the confidence threshold:

| Mode | When it wins | What it does |
|---|---|---|
| `offset` | The AD is a clean shift of the video (same cut, different start time) | Single global offset via normalised cross-correlation on downsampled raw audio |
| `drift` | Both tracks match but sample rates or clocks differ slightly | Measures offset at several points, fits a weighted linear drift model |
| `warp` *(default fallback)* | Cuts differ, inserted scenes, missing recaps, shifted ad breaks | Builds a top-K candidate lattice per window, runs a DP decoder with jump/curvature penalties + speech-rich bonuses, then fits a shape-preserving PCHIP warp and renders the output from a continuous time-map |
| `piecewise` | The classic stitch-and-crossfade approach | Anchor search + piecewise map with crossfades. Kept around for the easy cases and for comparison |

Warp mode is the default when offset/drift aren't enough, since solving the time-map globally tends to hold together better than reconciling independent per-chunk decisions after the fact.

## Installation

Requirements:

- Python 3.10+
- FFmpeg and ffprobe on your `PATH` ([ffmpeg.org](https://ffmpeg.org))

Then:

```bash
git clone https://github.com/JohnnyTheCoder1/ADSync.git
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

Open the resulting MKV in VLC, MPV, Plex, or Jellyfin and select the "Audio Description" track.

### CUDA acceleration

For an NVIDIA GPU, install the optional CUDA backend:

```bash
pip install -e ".[cuda]"
adsync devices
adsync sync episode.mkv ad_track.m4a --device cuda
```

The CUDA extra installs CuPy and its runtime libraries using CuPy's
[documented installation method](https://docs.cupy.dev/en/stable/install.html).
A compatible NVIDIA driver is required. The standard installation still works on the CPU.

`--device auto` is the default: larger correlations use CUDA when available,
while small jobs stay on the CPU. If CUDA fails to initialize or runs out of
memory, auto mode logs the reason and continues on the CPU. Use `--device cuda`
to require GPU processing or `--device cpu` to disable it. The option works with
`sync`, `analyze`, and `debug`; `ADSYNC_DEVICE` sets the default.

CUDA speeds up the FFT cross-correlations used to find offsets, measure drift,
and build the warp candidate lattice. Feature extraction, fingerprinting,
audio rendering, and encoding still run on the CPU, so the overall speedup
varies by source. Reports include the selected backend and counts of GPU and
CPU correlations.

### Shared-folder output

Outputs can go to a local folder, a mapped drive, or a network share. That is
useful when processing on one computer and keeping a Jellyfin library on
another, such as a laptop or home server.

```powershell
adsync sync episode.mkv ad_track.m4a --output-dir '\\SERVER\Media\TV'

# Set a default destination for this terminal session:
$env:ADSYNC_OUTPUT_DIR = '\\SERVER\Media\TV'
adsync sync episode.mkv ad_track.m4a
```

An explicit `-o` filename takes precedence over `--output-dir`, then
`ADSYNC_OUTPUT_DIR`. Without a destination setting, output stays beside the
source. The folder option also works with `mux` and `prep`.

ADSync finishes encoding before publishing the output. Copies to another
volume use a temporary non-media filename, then rename it when the transfer
finishes, keeping incomplete files out of media-library scans. Existing output
is replaced only after publication succeeds. If the destination becomes
unavailable, ADSync retains the completed file and reports its recovery path.
A failed `sync` publication also attempts to save its report beside that file.

The account running ADSync needs write access to the destination. Share
creation, authentication, and media-server library setup are managed separately.

## Commands

### Process a season, or several seasons

Point the season command at the video and AD folders for one show. It scans
subfolders, pairs episodes, and writes results into `Season 01`, `Season 02`,
and so on beneath the output folder.

```bash
# Check the pairings first, without processing media.
adsync season "Show videos" "AD tracks" --dry-run

# Prepare and sync every matched episode.
adsync season "Show videos" "AD tracks" --output-dir "Finished" --language eng

# Select seasons or individual episode numbers.
adsync season "Show videos" "AD tracks" --season 2 --season 3 --output-dir "Finished"
adsync season "Show videos" "AD tracks" --season 2 --episode 4 --episode 5
```

Matching uses identifiers such as `S02E04`, `[S02.E04]`, and `2x04`, with season
folders supplying context for simpler names. When both folders have a clear
matching number sequence, names such as `episode 1` and `ad 1` can also work.
One folder's known season can supply the other's missing season when at least
two unique episode numbers agree and there is no competing season.
The preview explains inferred season numbers. If no season is identified,
numeric inference uses Season 01; use `--season N` to supply the actual season.
Files are matched by identifiers, never by their position in a sorted list.
Duplicates, conflicting identifiers, and missing counterparts are listed for
review. Use `--strict` to stop before processing when any selected input is
unresolved.

By default, each output keeps the requested original language, downmixes
multichannel audio to stereo where needed, and adds the synced AD track.
Preparation happens in the final mux, saving a full intermediate video copy.
Video, subtitles, and attachments are copied. Already mono/stereo original
audio is copied too. Use `--no-prep` to retain all original audio streams.

The command detects usable CPU cores, available RAM, and working CUDA/VRAM,
then chooses concurrent episode jobs and per-job thread budgets. It estimates
memory from the selected media's duration, channels, and sample rate, and
reserves capacity for the system. These are planning estimates, not a guarantee
against memory pressure from other programs. Automatic concurrency is capped
at four jobs to limit competing media I/O. `--jobs N`, `--threads N`, and
`--device auto|cpu|cuda` provide control when tuning for a particular machine.
`adsync devices` shows the detected resources.

Rerunning the same command resumes unchanged completed episodes. Resume checks
source content, processing settings, algorithm/QC revisions, and media/report/QC hashes;
an existing MKV alone does not count as a completed job. Outputs without a
matching completion record require `--overwrite`. Each episode gets its own
log and report, and `season-summary.json` records the overall result. These
live in a persistent local cache, whose path is printed at startup; use
`--state-dir` to choose another location. A failed episode does not stop the
remaining jobs, and low-confidence results are marked for review.

The season command uses the same shared-folder publication and recovery path
as `sync`. Exit codes are `0` for successful or resumed results, `1` when review
is needed, `2` for failures or conflicts, and `130` for interruption.

To compare worker counts on representative material, run
`tools/benchmark_season.py --video-dir VIDEOS --ad-dir AD --workdir BENCHMARKS`.
It measures complete processing with one, two, and four workers, including
startup, rendering, and publication, and writes separate outputs for each run.

### Single-file commands

```bash
# Prep a source first (optional), keep one language's audio as stereo, drop the rest.
# Video/subs are stream-copied; already-stereo tracks are copied too (remux speed).
# Multichannel tracks get a dialog-forward downmix + true-peak limiter (libopus,
# ~4x faster than the aac encoder; pass --codec aac if you need AAC).
adsync prep episode.mkv --language eng

# Full sync, produces the final MKV with AD embedded
adsync sync episode.mkv ad_track.m4a -o episode.synced.mkv

# Analysis only, writes a JSON report, no output file
adsync analyze episode.mkv ad_track.m4a --report report.json

# Debug, dumps feature CSVs, anchor plots, and intermediate WAVs
adsync debug episode.mkv ad_track.m4a --workdir debug_out

# Mux only, you already have a synced AD file, just drop it in the container
adsync mux episode.mkv already_synced_ad.m4a -o final.mkv
```

## Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, or `cuda`; also configurable through `ADSYNC_DEVICE` |
| `--output-dir` | beside source | Local or shared destination folder; `ADSYNC_OUTPUT_DIR` sets the default |
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
| `--warp-search-radius` | auto | Per-window search radius (s) around the detected offset; auto shrinks to fingerprint span hints, else scales with the duration gap and offset scatter |
| `--no-fingerprint` | off | Skip landmark fingerprinting (span detection, bad-spot warnings, per-window warp hints) |
| `--no-speed-detect` | off | Skip automatic PAL-style speed detection and correction |
| `--no-fp-anchor` | off | Never anchor windows on fingerprint matches when correlation starves (differing mixes) |
| `--no-multiband` | off | Correlate full-band only, instead of three weighted bands with the narration band turned down |

Run `adsync <command> --help` for the full list.

## Tested on real material

This started on episodes of *From* (the MGM+ horror/mystery series), pairing fan-contributed AD tracks with retail releases. Sync held up on everything, including one episode where the AD and video had a real edit discontinuity mid-way through. Warp mode handled that as a single offset jump instead of trying to force a linear drift fit across it.

Since then it's been through a stack of feature films, and the two that fought hardest each turned into a feature. An AD rip of *The Exorcist* ran about 4% fast with the pitch up, a PAL transfer, which is where automatic speed detection came from. *Sinister* has a credits song that also plays much earlier in the film, and those repeated landmarks kept voting a stretch of the AD to a spot fifteen minutes before where it belonged, which is where the repeated-content filtering came from. Both sync clean now.

If you have the tracks, it works.

## Verification and difficult material

Season processing verifies the local render before publishing it. It checks content identity across the episode, targets edits and uncertain regions, preserves timing evidence and fitted coefficients in the report, and compares original/output video packet hashes. Inconclusive episodes remain available for review in local staging. A high overall alignment score alone is insufficient for publication.

Short endings require independent matching evidence. Sparse soundtrack matches, narration-heavy passages and different source edits can still need review. Reports retain the unresolved regions instead of silently treating them as synchronized. See [season processing](docs/season-processing.md) for verification receipts, content-bound caches and configurable local staging.

## Measured accuracy

`tools/accuracy_harness.py` applies edits with exactly known time maps (cuts up to 45 s, insertions, offsets, 200 ppm clock drift, a PAL-speed transfer) to real movie audio and scores every reported anchor against ground truth. Across the nine scenarios, CPU and CUDA produce matching scores: **median placement error ≤ 11 ms (typically ~2.5 ms), p95 ≤ 16 ms, at least 99% of anchors within 50 ms**. Individual anchors can still be substantially wrong around cuts or stretches without matching evidence, even when the overall score is high. Check the report's unmatched-content and bridged-region warnings. The harness is the regression check for alignment changes; pass `--device cpu` or `--device cuda` to compare backends.

## Roadmap

- [x] Eliminate the transient pitch drift during warped stretching (WSOLA or phase-vocoder resampler)
- [x] Audio landmark fingerprinting: global offset-span detection, edit localization, and bad-spot (unmatched content) warnings
- [x] Automatic detection and correction of PAL-style speed transfers
- [x] Repeated-content defenses: monotone-playback filtering, excursion vetting, and post-fit verification against fingerprint matches
- [x] Optional CUDA cross-correlation with CPU fallback and backend reporting
- [x] Shared-folder output with staged publication and recovery after transfer failures
- [x] Season processing with checked episode matching, automatic resource limits, and resume
- [ ] Precomputed AD offset database / cache
- [ ] Web UI for non-technical users
- [ ] Automatic detection of matching AD tracks from a library

## Contributing

Issues and PRs welcome. A few things that are especially useful:

- Bug reports with a JSON report attached (from `adsync analyze`).
- Hard cases. Tracks that come out wrong, even if all you can do is describe the source material rather than share it.
- Improvements to the warp decoder penalties, the confidence model, or the rendering path.

One change per PR if you can, makes review fast.

## License

MIT. See [LICENSE](LICENSE).

## Credits and a note on content

ADSync doesn't ship, download, or distribute any copyrighted video or audio. It's a local tool that runs against files you already have.

The AD tracks themselves mostly come from [audiovault.net](https://audiovault.net), which hosts a huge library of both real professionally-produced audio description (the tracks originally aired by networks and studios) and community-recorded ones. None of this workflow exists without them. If ADSync is useful to you, go support audiovault, and credit the people who recorded the tracks where you can.
