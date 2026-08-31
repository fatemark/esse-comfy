# esse_h3_loop — perfect AV loops for MiniMax H3

Custom ComfyUI nodes that make MiniMax H3 (open-weights FL2VA/T2VA) produce
**seamless loops** — picture *and* sound — instead of the almost-loops you
get from feeding the same image as first and last frame.

## Why same-frame FL2VA can't loop

A perfect loop needs two things at the wrap: the same pixels (position)
and the same motion (velocity), plus the same audio phase. FL2VA endpoint
conditioning only constrains position — and only softly, since the
generated last frame merely approximates the anchor while color drifts
across the clip. Nothing in training ever showed the model wrap-around
dynamics, and at inference the last frames never attend to the first
frames *as their successors*, so a "perfect loop" prompt has no mechanism
to act on. You get motion that decelerates into the anchor or arrives
with the wrong direction: a hitch, not a loop.

Both routes below fix the actual problem — they give the model real
context **across** the seam.

## Route 1 — Loop Bridge (recommended, works with image-anchored clips)

Generate clip A normally (fl2va from your image, or t2va). Then generate a
**bridge** whose head is pinned to A's *last* frames and whose tail is
pinned to A's *first* frames — both sliced straight out of A's sampler
latent (never decoded and re-encoded, so there is no color drift to
match away; sound is sliced and pinned too). The bridge's free middle
flows out of A's ending motion and arrives at A's opening motion: the
wrap itself is generated, with velocity continuity on both sides.
Splice A + middle and it loops back to A's frame 0.

Pinning uses the mechanisms ComfyUI's H3 integration ships natively:

- `conditioning_rows` (default): never-denoised keyframe cond rows at
  explicit frame indices — the same mechanism the H3 Motion Context
  chaining packs use in production for seamless clip joins. Requires
  ComfyUI ≥ 0.34 (earlier layouts reject mid-clip anchors; the node
  checks and refuses with a clear message).
- `latent_mask`: the pinned content is written into the bridge latent
  under a zero noise mask; H3 relabels preserved rows at the cond
  timestep and injects them clean every step, so the pinned spans come
  back bit-identical and the splice cuts are exact. Leaner sequence,
  less field mileage — try both.

Nodes: **H3 Loop Bridge** → sample → **H3 Loop Splice**.
Workflow: `example_workflows/h3_perfect_loop_bridge.json` — one queue
press renders A, the bridge, and the spliced loop (it's a straight DAG,
no cross-run latent saving needed).

## Route 2 — Mobius rotation (experimental, t2va only)

Training-free seamless looping per [Mobius (arXiv:2502.20307)](https://arxiv.org/abs/2502.20307):
at every denoising step the latent is rotated along time by a
step-dependent offset before the model sees it, and the prediction is
rotated back after. The wrap point lands somewhere else at every step, so
every frame is denoised with context from across the seam and the clip
comes out **cyclic by construction**.

The H3 adaptation handles what a generic implementation would get wrong:

- Rotation only in multiples of **5 latent steps = 17 frames**: H3's
  video latent covers frames in a 1-4-4-4-4 cycle, so any other offset
  would desync content spans from the fixed RoPE lattice (temporal
  warping, not translation).
- The **audio latent rotates in sync** on its own 40 Hz clock
  (17/3 audio steps per video step — exact every third offset, within
  ~8 ms otherwise; `exact_av` restricts to 51-frame offsets if you need
  sample-exact sync during sampling, e.g. for lipsync-ish content).
- A 17g+5-frame clip is 5g+2 latent steps: the 2 leading steps cannot
  join the cycle, so the loop is the clip **minus its first 5 frames**
  (which conveniently also absorb the video VAE's cold start — the wrap
  lands on a warm-decoded frame). **H3 Mobius Loop Trim** cuts them and
  the matching audio.
- Implemented as a `DIFFUSION_MODEL` wrapper, not a custom sampler: the
  sampler's state stays canonical, so it composes with res_multistep /
  euler / anything stock, CFG calls at one sigma share one offset, and
  previews stay upright. Offsets resolve statelessly from the sigma's
  position in the sampler's published schedule.

Rules: **text-to-AV only** (the patch refuses keyframes/references —
anchored content must not rotate; use the Bridge for that). Use lengths
**56 / 107 / 158 / 209 / 260 / 311** — every third point of the 17g+5
grid, where the audio cycle closes exactly. This route follows the paper
faithfully but is the one that needs empirical tuning on real renders
(`shift_frames`, step count); treat the Bridge as the proven default.

Nodes: **H3 Mobius Loop Patch** (on the model) → sample → **H3 Mobius
Loop Trim**. Workflow: `example_workflows/h3_perfect_loop_mobius.json`.

## Utility

**H3 Loop Seam Preview** rolls a finished loop by half so the wrap plays
mid-clip — a seam at a player's timeline ends is invisible unless the
player loops gaplessly, which preview players don't. Judge the loop on
the rolled render; ship the unrolled one.

## Install

Copy (or symlink) the `esse_h3_loop` folder into `ComfyUI/custom_nodes/`
on the machine that runs H3, and restart. No extra dependencies. Needs a
ComfyUI new enough to run H3 open weights natively, and ≥ 0.34 for the
bridge's `conditioning_rows` mode (checked at runtime).

Sanity-check the math anywhere with:

```bash
python3 esse_h3_loop/tests/run_tests.py
```

(74 checks over the grid arithmetic, both pin modes, the splice, the
rotation wrapper, and the trims — no ComfyUI or models needed.)

## Practical notes

- Keep the community-standard H3 sampling: **BasicGuider (CFG 1)**,
  res_multistep or euler, `ModelSamplingMiniMaxH3`. Turbo/distill LoRAs
  compose with both routes (the bridge is just conditioning; the Mobius
  wrapper is sampler-agnostic).
- Bridge lengths: source and bridge must share resolution; both lengths
  live on H3's 17g+5 grid (the stock nodes snap for you). Default
  124-frame bridge with 22-frame windows leaves an 80-frame free middle
  (~3.3 s) — give the transition at least ~2 s.
- If a faint shimmer survives a `conditioning_rows` splice, set
  `video_blend_frames` 3–6: the blend uses the bridge's own regenerated
  copies of the pinned spans, so it's a no-op in `latent_mask` mode and
  gentle everywhere else. `audio_crossfade_ms` 10 kills clicks the same
  way.
- Audio is cut at the same instants as the picture and normalized to
  frames/fps per piece, so H3's ±8 ms per-clip audio-grid overhang can't
  accumulate into an audible offset at the wrap (same fix the Motion
  Context pack applies to chains).
- **About loop LoRAs**: training one on genuine seamless loops is now
  possible (H3 is open-weight, AI-Toolkit/fal support it) and would *not*
  inherit the base model's defect — that defect is missing supervision,
  not architecture. But a LoRA is a statistical fix and composes with,
  rather than replaces, these nodes: it would make bridges shorter and
  Mobius renders converge faster, and you'd still trim/splice.

## Credits / prior art

- Pinning mechanics follow what the
  [H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
  chaining pack proved out for seamless clip joins (per-step keyframe
  rows at real pixel offsets, latent-sliced context, audio grid
  overhang compensation). This pack generalizes the idea from
  *continuation* to *closure*: pin both ends, aim the tail at the
  clip's own opening.
- Rotation route: [Mobius: Text to Seamless Looping Video Generation
  via Latent Shift](https://arxiv.org/abs/2502.20307), adapted to H3's
  1-4-4-4-4 frame cycle and joint AV latents.
