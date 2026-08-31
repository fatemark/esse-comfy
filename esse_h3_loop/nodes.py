"""Perfect-loop nodes for MiniMax H3 in ComfyUI.

Two independent routes to a seamless AV loop, sharing one insight: a loop
fails at the seam not because the endpoint pixels differ but because the
motion (and sound) across the wrap has no reason to be continuous. FL2VA
with the same image at both ends pins position, not velocity - the model
never saw wrap-around dynamics in training and cannot be prompted into
them. Both routes below give the model real cross-seam context instead.

ROUTE 1 - loop closure by latent-handoff bridging (robust, image-anchorable)
    Generate clip A normally. Then generate a BRIDGE whose head is pinned
    to A's last frames and whose tail is pinned to A's FIRST frames - both
    sliced straight out of A's sampler latent, never decoded and re-encoded.
    The bridge's middle is free generation that flows out of A's ending
    motion and into A's opening motion. Splice A + middle: both joins are
    model-generated continuations, including the wrap itself.
    Pinning uses the mechanisms ComfyUI's H3 integration ships with:
      conditioning_rows  never-denoised keyframe cond rows at explicit
                         frame indices (what the H3 Motion Context chaining
                         pack uses in production for seamless joins).
      latent_mask        the content written into the bridge latent with a
                         zero noise mask; H3 relabels preserved rows at the
                         cond timestep and injects them clean every step,
                         so the pinned spans come back bit-identical.

ROUTE 2 - Mobius latent rotation (training-free, t2va only)
    Following Mobius (arXiv:2502.20307): at every denoising step the model
    sees the latent rotated along time by a step-dependent offset and its
    prediction is rotated back, so the wrap point lands at a different
    position each step and every frame is denoised with cross-seam context.
    The loop is cyclic by construction rather than closed after the fact.
    H3 specifics handled here: rotation must be a multiple of 5 latent
    steps (the 1-4-4-4-4 frame cycle, so content spans keep matching the
    fixed RoPE lattice), the audio latent rotates in sync on its own 40 Hz
    clock, and the 2 leading latent steps stay static (a 17g+5 clip is
    5g+2 steps - only the trailing 5g steps form a rotatable cycle), so
    the first 5 pixel frames are trimmed afterwards.

Both routes want the community-standard H3 sampling setup: BasicGuider
(no negative / CFG 1), res_multistep or euler, ModelSamplingMiniMaxH3.
"""

import logging
import math

import torch

import comfy.nested_tensor
import comfy.patcher_extension
import node_helpers

try:  # package-relative when loaded by ComfyUI, flat when loaded by tests
    from .grid import (AUDIO_HZ, CONTEXT_WINDOWS, CYCLE_FRAMES, CYCLE_STEPS,
                       FPS, FRAME_RESCALE, audio_overhang, audio_t_for_frames,
                       is_native_length, pixel_frames, snap_window_down,
                       step_offsets, steps_for_frames)
except ImportError:  # pragma: no cover
    from grid import (AUDIO_HZ, CONTEXT_WINDOWS, CYCLE_FRAMES, CYCLE_STEPS,
                      FPS, FRAME_RESCALE, audio_overhang, audio_t_for_frames,
                      is_native_length, pixel_frames, snap_window_down,
                      step_offsets, steps_for_frames)

_LOG = logging.getLogger("esse_h3_loop")

_CATEGORY = "MiniMax H3/Loop"


# ---------------------------------------------------------------------------
# environment check
# ---------------------------------------------------------------------------

_h3_state = None  # None = unchecked, dict = capabilities, str = hard failure


def _h3_capabilities():
    """Prove this ComfyUI has the H3 integration these nodes are built on,
    and report which generation of it.

    Hard requirements for every node here: comfy.ldm.minimax.model exists
    and the 1-4-4-4-4 latent frame cycle holds (all trim/phase arithmetic
    is built on it). Soft capability: `arbitrary_anchors` - whether
    PackedLayout takes keyframe anchors at arbitrary indices. The first
    H3 integration (still shipping in e.g. v0.30.x) raises "only
    first/last keyframe anchors are supported" and has no audio keyframe
    rows at all, detectable from its extra frame_count parameter; later
    ComfyUI lifted both. Only the bridge's conditioning_rows mode needs
    the lifted layout - latent_mask pinning and the Mobius rotation work
    on both generations. Checked once, on first use, so an
    installed-but-unused pack costs nothing at startup.
    """
    global _h3_state
    if isinstance(_h3_state, dict):
        return _h3_state
    if isinstance(_h3_state, str):
        raise RuntimeError(_h3_state)
    try:
        import comfy.ldm.minimax.model as mm
    except ImportError as exc:
        _h3_state = ("esse_h3_loop: this ComfyUI has no MiniMax H3 support "
                     "(comfy.ldm.minimax.model is missing). Update ComfyUI.")
        raise RuntimeError(_h3_state) from exc
    if tuple(getattr(mm, "FRAME_PER_TOKEN", ())) != (1, 4, 4, 4, 4):
        _h3_state = ("esse_h3_loop: ComfyUI's H3 FRAME_PER_TOKEN is %r, this "
                     "pack assumes (1, 4, 4, 4, 4). The latent grid moved; "
                     "refusing rather than splicing at wrong instants."
                     % (getattr(mm, "FRAME_PER_TOKEN", None),))
        raise RuntimeError(_h3_state)
    import inspect
    try:
        params = inspect.signature(mm.PackedLayout.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    _h3_state = {"arbitrary_anchors": "frame_count" not in params}
    if not _h3_state["arbitrary_anchors"]:
        _LOG.info("esse_h3_loop: this ComfyUI has the first-generation H3 "
                  "layout (first/last anchors only). latent_mask pinning "
                  "and the Mobius route work; conditioning_rows needs a "
                  "ComfyUI update.")
    return _h3_state


def _ensure_h3_ready():
    _h3_capabilities()


# ---------------------------------------------------------------------------
# AV latent helpers
# ---------------------------------------------------------------------------

def _streams_from_latent(latent):
    """Unpack an H3 AV latent dict into [video, audio] tensors.

    NestedTensor.__getitem__ broadcasts into every member rather than
    selecting one, so samples[0] would strip the batch dim off both
    streams; unbind() is the correct accessor. A plain list/tuple is also
    accepted (the Motion Context pack's latent loader produces one).
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        raise ValueError("esse_h3_loop: LATENT is missing 'samples'")
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "esse_h3_loop: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r. Wire the LATENT of an H3 node or "
            "sampler, not a plain video latent." % type(samples))
    if not parts:
        raise ValueError("esse_h3_loop: AV latent contains no streams")
    video = parts[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("esse_h3_loop: expected video latent [B,C,T,H,W], "
                         "got %s" % (tuple(video.shape),))
    audio = None
    if len(parts) > 1:
        audio = parts[1]
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        if audio.ndim != 4:
            raise ValueError("esse_h3_loop: expected audio latent [B,C,2,T], "
                             "got %s" % (tuple(audio.shape),))
    return video, audio


def _match_audio_len(waveform, sr, seconds, what=""):
    """Trim or zero-pad a [B,C,L] waveform to exactly `seconds`.

    H3 rounds its audio grid to the nearest 40 Hz step, so a decoded clip
    carries about 8ms more or less sound than picture depending on its
    length. Splicing without normalizing lets that error compound at every
    join; zero is the only honest fill for samples that were never
    generated (anything else fabricates content to hide a seam).
    """
    want = int(round(seconds * sr))
    have = int(waveform.shape[-1])
    if have > want:
        _LOG.info("esse_h3_loop: %s audio trimmed %d samples (%.2fms)",
                  what, have - want, (have - want) / sr * 1000.0)
        return waveform[..., :want]
    if have < want:
        _LOG.info("esse_h3_loop: %s audio padded %d samples (%.2fms)",
                  what, want - have, (want - have) / sr * 1000.0)
        return torch.nn.functional.pad(waveform, (0, want - have))
    return waveform


def _slice_audio_seconds(waveform, sr, t0, t1):
    a = int(round(t0 * sr))
    b = int(round(t1 * sr))
    return waveform[..., a:b]


def _crossfade(dst, src, rising):
    """Blend src into dst over their (equal) length. rising=True fades
    src in 0 -> 1 (dst survives at the start), rising=False fades it out.
    Cosine ramp that never quite reaches the pure endpoints, so no frame
    is exactly duplicated material."""
    n = dst.shape[0]
    w = 0.5 - 0.5 * torch.cos(
        torch.linspace(0, math.pi, n + 2, dtype=torch.float32)[1:-1])
    if not rising:
        w = w.flip(0)
    shape = [n] + [1] * (dst.ndim - 1)
    w = w.view(shape).to(dst.device, dst.dtype)
    return dst * (1.0 - w) + src * w


def _crossfade_wave(dst, src, rising):
    """Equal-power crossfade on [B,C,N] waveforms (audio wants constant
    energy through the blend, not constant amplitude)."""
    n = dst.shape[-1]
    t = torch.linspace(0, math.pi / 2.0, n + 2, dtype=torch.float32)[1:-1]
    w_in, w_out = torch.sin(t), torch.cos(t)
    a, b = (w_out, w_in) if rising else (w_in, w_out)
    a = a.view(1, 1, n).to(dst.device, dst.dtype)
    b = b.view(1, 1, n).to(dst.device, dst.dtype)
    return dst * a + src * b


# ---------------------------------------------------------------------------
# Route 1: the loop bridge
# ---------------------------------------------------------------------------

class EsseH3LoopBridge:
    """Condition a bridge clip that closes clip A into a perfect loop.

    The bridge's head is pinned to A's LAST frames and its tail to A's
    FIRST frames, both sliced straight out of A's sampler latent - exactly
    what the model produced, no h264 decode, no VAE round trip, no colour
    drift for a matcher to chase. The middle is free generation with real
    motion context on both sides, so it leaves A the way A actually ends
    and arrives at A's opening the way A actually begins. That second half
    is what same-image FL2VA cannot do: it is the wrap itself, generated.

    Phase safety: clips are 17g+5 frames = 5g+2 latent steps on a 1-4-4-4-4
    frames-per-step cycle. The offered windows (5/22/39/56) are the runs
    that are whole steps starting at cycle position 0, and both pin sites
    (bridge start; bridge end minus a window) land on multiples of the
    17-frame cycle, so sliced content and written positions always agree.
    Asserted, not assumed.

    Wire the conditioning from a MiniMax H3 Image to Video node WITHOUT
    first/last frames (its prompt describes the bridge motion), and its
    LATENT output as this node's latent. Sample with the usual H3 setup
    (BasicGuider, CFG 1). Then decode and feed EsseH3LoopSplice.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "From a MiniMax H3 Image to Video node with NO "
                               "first/last frame wired. Its prompt describes "
                               "the bridge's transition motion. Any keyframe "
                               "anchors inside the pinned spans are dropped."}),
                "latent": ("LATENT", {
                    "tooltip": "The SAME Image to Video node's LATENT output "
                               "(the bridge canvas). Width/height must match "
                               "the source clip; length sets bridge duration."}),
                "source_latent": ("LATENT", {
                    "tooltip": "Clip A's SAMPLER OUTPUT latent (the one you "
                               "also wire into the decode nodes). Supplies "
                               "picture and sound for both pins."}),
                "head_frames": (["22", "5", "39", "56"], {
                    "default": "22",
                    "tooltip": "Frames of A's ENDING pinned at the bridge "
                               "start. Only these lengths are whole latent "
                               "steps. 22 is the chaining sweet spot; 5 is "
                               "barely fluid; longer pins more motion but "
                               "shrinks the free middle."}),
                "tail_frames": (["22", "5", "39", "56"], {
                    "default": "22",
                    "tooltip": "Frames of A's OPENING pinned at the bridge "
                               "end - the wrap target. The middle must land "
                               "on A's first motion, not just its first "
                               "frame, which is what makes the loop close."}),
                "audio_head_frames": ("INT", {
                    "default": 24, "min": 0, "max": 240,
                    "tooltip": "Frames of A's tail AUDIO pinned at the bridge "
                               "start, end-aligned with the pinned picture. "
                               "0 follows head_frames. Multiples of 3 land on "
                               "the 40 Hz grid; 24 pins the last second."}),
                "audio_tail_frames": ("INT", {
                    "default": 24, "min": 0, "max": 240,
                    "tooltip": "Frames of A's opening AUDIO pinned at the "
                               "bridge end, start-aligned with the pinned "
                               "picture. 0 follows tail_frames; clamped to "
                               "tail_frames so it cannot reach back over the "
                               "generated middle."}),
                "pin_mode": (["latent_mask", "conditioning_rows"], {
                    "default": "latent_mask",
                    "tooltip": "latent_mask (recommended for loops): A's "
                               "content is written into the bridge latent "
                               "under a zero noise mask; H3 runs those rows "
                               "at the cond timestep and injects them clean "
                               "each step, so the pinned spans come back "
                               "latent-identical and the splice can cut "
                               "inside them for an exact wrap.\n"
                               "conditioning_rows: never-denoised keyframe "
                               "rows, the H3 chaining packs' mechanism. "
                               "Continuation from the head pin is seamless, "
                               "but ARRIVAL at the tail rows is soft - the "
                               "middle lands near A's opening, not on it "
                               "(the same softness as native fl2va last-"
                               "frame anchors), so the wrap can read as a "
                               "jump cut between similar shots."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "H3_LOOP_INFO", "STRING")
    RETURN_NAMES = ("conditioning", "latent", "loop_info", "report")
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Pin a bridge clip's head to clip A's ending and its tail "
                   "to A's opening, so the sampled middle closes A into a "
                   "seamless loop. Splice with H3 Loop Splice afterwards.")

    def build(self, conditioning, latent, source_latent, head_frames,
              tail_frames, audio_head_frames=24, audio_tail_frames=24,
              pin_mode="latent_mask"):
        caps = _h3_capabilities()
        if pin_mode == "conditioning_rows" and not caps["arbitrary_anchors"]:
            raise ValueError(
                "esse_h3_loop: this ComfyUI still has the first-generation "
                "H3 layout, which rejects keyframe anchors other than the "
                "first and last frame, so pin_mode=conditioning_rows cannot "
                "place a loop bridge on it. Either set pin_mode to "
                "'latent_mask' (works on this build), or update ComfyUI to "
                "get the lifted layout that the H3 chaining packs use.")
        head = int(head_frames)
        tail = int(tail_frames)

        b_video, b_audio = _streams_from_latent(latent)
        s_video, s_audio = _streams_from_latent(source_latent)
        if b_audio is None or s_audio is None:
            raise ValueError(
                "esse_h3_loop: both latents need video AND audio streams. "
                "Wire H3 AV latents (bridge: the Image to Video node's "
                "latent; source: clip A's sampler output).")

        T_b = int(b_video.shape[2])
        T_s = int(s_video.shape[2])
        F_b = pixel_frames(T_b)
        F_s = pixel_frames(T_s)
        if b_video.shape[3:] != s_video.shape[3:] or b_video.shape[1] != s_video.shape[1]:
            raise ValueError(
                "esse_h3_loop: bridge is %dx%d (%d ch) but source is %dx%d "
                "(%d ch). A latent cannot be resized; generate both at the "
                "same resolution."
                % (b_video.shape[4] * 16, b_video.shape[3] * 16, b_video.shape[1],
                   s_video.shape[4] * 16, s_video.shape[3] * 16, s_video.shape[1]))

        middle = F_b - head - tail
        if middle < CYCLE_FRAMES:
            raise ValueError(
                "esse_h3_loop: bridge of %d frames minus %d pinned head and "
                "%d pinned tail leaves %d free frames. Give the transition "
                "at least ~2s: raise the bridge length (124 = ~5s) or "
                "shorten the windows." % (F_b, head, tail, middle))

        steps_h = steps_for_frames(head)
        steps_t = steps_for_frames(tail)
        if steps_h is None or steps_t is None:  # widget list makes this unreachable
            raise ValueError("esse_h3_loop: context windows must be one of %s"
                             % (CONTEXT_WINDOWS,))
        if steps_h > T_s or steps_t > T_s:
            raise ValueError(
                "esse_h3_loop: source clip has %d frames; it cannot supply a "
                "%d frame window. Use a shorter window or a longer clip A."
                % (F_s, max(head, tail)))

        # Phase contract: sliced content must sit at positions with the same
        # 1-4-4-4-4 phase it was generated at, or the written coordinates
        # would disagree with the frames inside the blocks by up to 3 frames.
        if (T_s - steps_h) % CYCLE_STEPS != 0:
            raise RuntimeError(
                "esse_h3_loop: the %d-step tail of a %d-step source starts at "
                "cycle position %d, not 0. Source is not on H3's 17g+5 grid."
                % (steps_h, T_s, (T_s - steps_h) % CYCLE_STEPS))
        tail_pos = F_b - tail
        if tail_pos % CYCLE_FRAMES != 0:
            raise RuntimeError(
                "esse_h3_loop: bridge end minus the tail window lands at "
                "frame %d, off the 17-frame cycle. Bridge length %d or "
                "window %d is not on H3's grid." % (tail_pos, F_b, tail))

        # ---- audio windows -------------------------------------------------
        Ta_s = int(s_audio.shape[-1])
        Ta_b = int(b_audio.shape[-1])
        over = audio_overhang(F_s, Ta_s)
        if not (-0.5 < over < 0.5):
            _LOG.warning("esse_h3_loop: source audio grid is unexpected "
                         "(%d steps for %d frames); assuming no overhang.",
                         Ta_s, F_s)
            over = 0.0

        a_head = int(audio_head_frames) or head
        a_tail = int(audio_tail_frames) or tail
        if a_tail > tail:
            # The head-side window may reach back before the bridge starts
            # (negative coordinates are outside the generated timeline), but
            # a tail-side window longer than its picture span would reach
            # back OVER the generated middle and pin sound where the model
            # must stay free.
            _LOG.warning("esse_h3_loop: audio_tail_frames %d clamped to the "
                         "%d frame tail window.", a_tail, tail)
            a_tail = tail
        ra_h = min(int(round(a_head / FPS * AUDIO_HZ)), Ta_s)
        ra_t = min(int(round(a_tail * FRAME_RESCALE)), Ta_s)

        head_audio = s_audio[:1, ..., Ta_s - ra_h:].clone() if ra_h else None
        tail_audio = s_audio[:1, ..., :ra_t].clone() if ra_t else None

        info = {
            "kind": "bridge",
            "pin_mode": pin_mode,
            "head_span": head,
            "tail_span": tail,
            "bridge_frames": F_b,
            "source_frames": F_s,
            "fps": FPS,
        }

        if pin_mode == "latent_mask":
            out_latent = self._build_masked_latent(
                b_video, b_audio, s_video, s_audio, T_b, steps_h, steps_t,
                Ta_b, ra_h, ra_t, tail_pos)
            out_cond = self._scrub_keyframes(conditioning, head, tail, F_b)
        else:
            keyframes = self._build_keyframes(
                s_video, T_s, steps_h, steps_t, tail_pos, head_audio,
                tail_audio, ra_h, ra_t, head, over)
            out_cond = self._merge_keyframes(
                self._scrub_keyframes(conditioning, head, tail, F_b), keyframes)
            out_latent = latent

        report = (
            "loop bridge (%s): %d frame bridge = %d pinned head + %d free + "
            "%d pinned tail\n"
            "head <- source frames %d..%d (%d latent steps), tail <- source "
            "frames 0..%d at bridge frame %d\n"
            "audio: %d steps end-aligned at frame %d, %d steps start-aligned "
            "at frame %d (source overhang %+.2f step)\n"
            "splice: keep bridge frames %d..%d after clip A"
            % (pin_mode, F_b, head, middle, tail,
               F_s - head, F_s - 1, steps_h, tail - 1, tail_pos,
               ra_h, head, ra_t, tail_pos, over,
               head, F_b - tail - 1))
        _LOG.info("esse_h3_loop: %s", report.replace("\n", " | "))
        return (out_cond, out_latent, info, report)

    # -- conditioning_rows mechanics ----------------------------------------

    @staticmethod
    def _build_keyframes(s_video, T_s, steps_h, steps_t, tail_pos,
                         head_audio, tail_audio, ra_h, ra_t, head_span, over):
        """One cond block per latent step, each at its real pixel offset -
        the layout places a 1-step latent at FRAME_RESCALE * index past the
        timeline origin, so per-step placement keeps content and coordinate
        in agreement whatever the step's frame span is."""
        keyframes = []
        offs_h = step_offsets(steps_h)
        start = T_s - steps_h
        for k in range(steps_h):
            keyframes.append({
                "resolved_frame_index": offs_h[k],
                "latent": s_video[:1, :, start + k:start + k + 1].clone(),
            })
        offs_t = step_offsets(steps_t)
        for k in range(steps_t):
            keyframes.append({
                "resolved_frame_index": tail_pos + offs_t[k],
                "latent": s_video[:1, :, k:k + 1].clone(),
            })
        if head_audio is not None:
            # End-align with the pinned picture: both are A's tail, so both
            # must end at the same instant. The sliced audio overshoots A's
            # last frame by `over` of a step (H3 rounds its grid to the
            # nearest step), so the end coordinate moves by exactly that,
            # then snaps onto the bridge's own 40 Hz grid. The window index
            # is its START, which is fractional and usually negative -
            # legal arithmetic in the layout since ComfyUI 0.34.
            end_frame = head_span + over / FRAME_RESCALE
            end_frame = round(FRAME_RESCALE * end_frame) / FRAME_RESCALE
            keyframes.append({
                "resolved_frame_index": end_frame - ra_h / FRAME_RESCALE,
                "audio_latent": head_audio,
            })
        if tail_audio is not None:
            # Start-align: A's audio step 0 sits exactly on A's frame 0, and
            # A's frame 0 maps to the bridge's tail_pos, an integer index.
            # The window may fall a third of a step short of the bridge end;
            # that sliver is where the real clip A takes over anyway.
            keyframes.append({
                "resolved_frame_index": tail_pos,
                "audio_latent": tail_audio,
            })
        return keyframes

    @staticmethod
    def _scrub_keyframes(conditioning, head, tail, F_b):
        """Drop upstream keyframe anchors that fall inside the pinned spans
        (they would fight content the pins already decide). Anchors in the
        free middle are a legitimate user choice and are kept."""
        out, dropped = [], []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = d.get("minimax_keyframes") or []
            kept = []
            for kf in prior:
                p = kf.get("resolved_frame_index", 0)
                if p < head or p >= F_b - tail:
                    dropped.append(p)
                    continue
                kept.append(dict(kf))
            if kept:
                d["minimax_keyframes"] = kept
            else:
                d.pop("minimax_keyframes", None)
            out.append([emb, d])
        if dropped:
            _LOG.warning(
                "esse_h3_loop: dropped %d upstream keyframe anchor(s) at "
                "frame(s) %s - the loop pins already decide those spans. "
                "Wire the bridge conditioning from an Image to Video node "
                "with no first/last frame.", len(dropped), sorted(set(dropped)))
        return out

    @staticmethod
    def _merge_keyframes(conditioning, keyframes):
        out = []
        for emb, extra in conditioning:
            d = extra.copy()
            d["minimax_keyframes"] = (d.get("minimax_keyframes") or []) + keyframes
            out.append([emb, d])
        return out

    # -- latent_mask mechanics ----------------------------------------------

    @staticmethod
    def _build_masked_latent(b_video, b_audio, s_video, s_audio, T_b,
                             steps_h, steps_t, Ta_b, ra_h, ra_t, tail_pos):
        """Write the pinned spans into the bridge latent and zero their
        noise mask. H3's integration relabels preserved rows at the cond
        timestep and re-injects them clean at every step, so this is the
        keyframe mechanism minus the duplicate rows - and the pinned spans
        come back bit-identical, which makes the splice cuts exact."""
        video = torch.zeros_like(b_video[:1])
        video[:, :, :steps_h] = s_video[:1, :, s_video.shape[2] - steps_h:].to(
            video.device, video.dtype)
        video[:, :, T_b - steps_t:] = s_video[:1, :, :steps_t].to(
            video.device, video.dtype)
        vmask = torch.ones((1, b_video.shape[1], T_b,
                            b_video.shape[3], b_video.shape[4]),
                           dtype=torch.float32, device=video.device)
        vmask[:, :, :steps_h] = 0.0
        vmask[:, :, T_b - steps_t:] = 0.0

        audio = torch.zeros_like(b_audio[:1])
        amask = torch.ones((1, b_audio.shape[1], 2, Ta_b), dtype=torch.float32,
                           device=audio.device)
        if ra_h:
            n = min(ra_h, Ta_b)
            audio[..., :n] = s_audio[:1, ..., s_audio.shape[-1] - ra_h:
                                     s_audio.shape[-1] - ra_h + n].to(
                audio.device, audio.dtype)
            amask[..., :n] = 0.0
        if ra_t:
            # start the pinned window on the bridge's own audio grid step
            # nearest to the tail picture's start; sub-step placement error
            # is at most a third of a step (~8ms), and the splice cuts at
            # exact pixel time regardless.
            a0 = min(max(0, int(round(tail_pos * FRAME_RESCALE))), Ta_b)
            n = min(ra_t, Ta_b - a0)
            if n > 0:
                audio[..., a0:a0 + n] = s_audio[:1, ..., :n].to(
                    audio.device, audio.dtype)
                amask[..., a0:] = 0.0
        return {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((vmask, amask)),
        }


class EsseH3LoopSplice:
    """Cut the bridge's pinned spans and splice the free middle after clip A.

    The output plays A in full, then the bridge middle, then wraps to A's
    first frame - which is exactly where the bridge middle was generated to
    arrive. Both joins are cuts between content the model generated in one
    anothers' context, the same cut the H3 chaining packs make.

    Audio is cut at the same instants and normalized to frames/fps at each
    piece, so H3's ~8ms per-clip audio-grid overhang cannot accumulate into
    an audible offset at the wrap. Optional crossfades blend each join with
    the material the cut discards (the bridge's regenerated copies of A),
    which is a no-op when that material is bit-identical (latent_mask mode)
    and belt-and-braces when it is a regeneration (conditioning_rows mode).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE", {"tooltip": "Decoded frames of clip A."}),
                "images_bridge": ("IMAGE", {"tooltip": "Decoded frames of the bridge."}),
                "loop_info": ("H3_LOOP_INFO",),
            },
            "optional": {
                "audio_a": ("AUDIO",),
                "audio_bridge": ("AUDIO",),
                "video_blend_frames": ("INT", {
                    "default": 0, "min": 0, "max": 16,
                    "tooltip": "Frames of crossfade at each join, blended "
                               "with the bridge's own copy of that span. 0 "
                               "is a clean cut (recommended first try); 3-6 "
                               "softens any residual shimmer in "
                               "conditioning_rows mode."}),
                "audio_crossfade_ms": ("INT", {
                    "default": 10, "min": 0, "max": 500,
                    "tooltip": "Equal-power crossfade at each audio join, "
                               "using the sound the cut discards. Kills "
                               "clicks; 0 for a hard cut."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                  "step": 0.001}),
                # appended last: ComfyUI stores widget values positionally,
                # so inserting this next to the other knobs would shift fps
                # in every workflow saved against the older node
                "seam_inset_frames": ("INT", {
                    "default": 5, "min": 0, "max": 16,
                    "tooltip": "Cut both joins this many frames INSIDE the "
                               "pinned spans instead of exactly at their "
                               "edges. In latent_mask mode the frames on "
                               "either side of each cut then decode from "
                               "identical latent content, so the wrap seam "
                               "is exact by construction; any residual "
                               "softness stays at the mask edge, where the "
                               "sampler already smoothed it during "
                               "generation. The loop starts at clip A frame "
                               "N instead of 0 (irrelevant for a loop). 0 "
                               "restores edge cuts."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "splice"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Splice clip A and the loop bridge's free middle into the "
                   "final seamless loop, picture and sound cut at the same "
                   "instants.")

    def splice(self, images_a, images_bridge, loop_info, audio_a=None,
               audio_bridge=None, video_blend_frames=0, audio_crossfade_ms=10,
               fps=24.0, seam_inset_frames=5):
        if loop_info.get("kind") != "bridge":
            raise ValueError("esse_h3_loop: loop_info is not from "
                             "EsseH3LoopBridge")
        head = int(loop_info["head_span"])
        tail = int(loop_info["tail_span"])
        F_b = int(loop_info["bridge_frames"])
        F_a = int(loop_info["source_frames"])
        mode = loop_info.get("pin_mode", "conditioning_rows")

        if int(images_bridge.shape[0]) != F_b:
            raise ValueError(
                "esse_h3_loop: bridge decodes to %d frames but the loop was "
                "built for %d. Wire the decoded frames of the same bridge "
                "generation." % (int(images_bridge.shape[0]), F_b))
        if int(images_a.shape[0]) != F_a:
            _LOG.warning("esse_h3_loop: clip A has %d frames, the loop was "
                         "built against %d. Splicing what is wired.",
                         int(images_a.shape[0]), F_a)
            F_a = int(images_a.shape[0])

        # Cut both joins `inset` frames INSIDE the pinned spans rather than
        # at their edges. The kept bridge segment then begins and ends with
        # pinned copies of A, so each cut sits between a real A frame and
        # its pinned counterpart - identical latent content in latent_mask
        # mode, so the wrap is exact by construction. The mask-edge
        # transition (free middle meeting pinned content) stays inside the
        # segment, where the sampler smoothed it during generation. Head
        # pins continue motion and bind hard; tail rows are arrived at
        # softly (native fl2va's known last-frame behaviour), which is why
        # an edge cut at the wrap can land near A's opening instead of on
        # it - the inset is what removes that residual.
        inset = max(0, int(seam_inset_frames))
        inset = min(inset, head - 1, tail - 1, (F_a - 1) // 2)
        a_lo, a_hi = inset, F_a - inset
        b_lo, b_hi = head - inset, F_b - tail + inset

        a = images_a[a_lo:a_hi]
        mid = images_bridge[b_lo:b_hi]

        blend = int(video_blend_frames)
        blend = min(blend, b_lo, tail - inset, a.shape[0] // 2, mid.shape[0] // 2)
        if blend > 0:
            a = a.clone()
            # join 1: A's ending eases into the bridge's copy of it
            a[-blend:] = _crossfade(
                a[-blend:], images_bridge[b_lo - blend:b_lo].to(a.device),
                rising=True)
            # the wrap: the bridge's copy of A's opening eases into A
            a[:blend] = _crossfade(
                images_bridge[b_hi:b_hi + blend].to(a.device),
                a[:blend], rising=True)
        loop = torch.cat([a, mid.to(a.device)], dim=0)

        out_audio = None
        if audio_a is not None and audio_bridge is not None:
            sr = int(audio_a["sample_rate"])
            if int(audio_bridge["sample_rate"]) != sr:
                raise ValueError("esse_h3_loop: audio sample rates differ "
                                 "(%d vs %d)" % (sr, int(audio_bridge["sample_rate"])))
            wav_a = _match_audio_len(
                _slice_audio_seconds(audio_a["waveform"], sr, a_lo / fps,
                                     F_a / fps),
                sr, a.shape[0] / fps, "clip A")
            wav_b = audio_bridge["waveform"]
            mid_wav = _match_audio_len(
                _slice_audio_seconds(wav_b, sr, b_lo / fps, b_hi / fps),
                sr, mid.shape[0] / fps, "bridge middle")

            j = int(round(audio_crossfade_ms / 1000.0 * sr))
            j = min(j, wav_a.shape[-1] // 2, mid_wav.shape[-1] // 2,
                    int(round(b_lo / fps * sr)),
                    int(round((F_b - b_hi) / fps * sr)))
            if j > 0:
                wav_a = wav_a.clone()
                # join 1: fade A's last j samples into the discarded sound
                # just before the kept bridge segment starts
                pre = _slice_audio_seconds(wav_b, sr, b_lo / fps - j / sr,
                                           b_lo / fps)
                if pre.shape[-1] == j:
                    wav_a[..., -j:] = _crossfade_wave(
                        wav_a[..., -j:], pre.to(wav_a.device), rising=True)
                # the wrap: the discarded sound just after the kept segment
                # ends fades into the loop's opening
                post = _slice_audio_seconds(wav_b, sr, b_hi / fps,
                                            b_hi / fps + j / sr)
                if post.shape[-1] == j:
                    wav_a[..., :j] = _crossfade_wave(
                        post.to(wav_a.device), wav_a[..., :j], rising=True)
            out_audio = {"waveform": torch.cat([wav_a, mid_wav.to(wav_a.device)], dim=-1),
                         "sample_rate": sr}
        elif audio_a is not None or audio_bridge is not None:
            _LOG.warning("esse_h3_loop: only one of audio_a/audio_bridge is "
                         "wired; the loop is silent. Wire both or neither.")

        # Seam quality, measured: frame deltas at the two cuts against the
        # loop's own typical frame-to-frame delta. A seam near 1.0x baseline
        # is invisible; several times baseline reads as a jump cut.
        def _delta(x, y):
            return float((x.float() - y.float()).abs().mean())

        pairs = [_delta(loop[i], loop[i + 1]) for i in range(0, loop.shape[0] - 1, 8)]
        base = max(sum(pairs) / max(len(pairs), 1), 1e-6)
        join1 = _delta(loop[a.shape[0] - 1], loop[a.shape[0]])
        wrap = _delta(loop[-1], loop[0])

        report = ("loop: %d frames (%.2fs at %.3g fps) = clip A[%d:%d] + "
                  "bridge[%d:%d] (%s, seam inset %d)\n"
                  "seam deltas vs baseline %.4f: join %.4f (%.1fx), wrap "
                  "%.4f (%.1fx) - near 1x is seamless"
                  % (loop.shape[0], loop.shape[0] / fps, fps, a_lo, a_hi,
                     b_lo, b_hi, mode, inset,
                     base, join1, join1 / base, wrap, wrap / base))
        _LOG.info("esse_h3_loop: %s", report.replace("\n", " | "))
        return (loop, out_audio, report)


# ---------------------------------------------------------------------------
# Route 2: Mobius rotation
# ---------------------------------------------------------------------------

class _MobiusRotationWrapper:
    """DIFFUSION_MODEL wrapper: rotate the AV latent before each model call,
    rotate the prediction back after. The sampler's state stays canonical,
    so this composes with any stock sampler (res_multistep included), CFG
    calls at one sigma share one offset, and previews stay upright.

    Offsets advance per sampler STEP, resolved statelessly from the sigma's
    position in transformer_options["sample_sigmas"] (the same lookup H3's
    own PDD heads use); a tiny sigma-transition counter covers samplers
    that do not publish their schedule.
    """

    def __init__(self, shift_steps, exact_av):
        self.shift_steps = int(shift_steps)
        self.exact_av = bool(exact_av)
        self._last_sigma = None
        self._counter = 0

    def _step_index(self, sigma, transformer_options):
        ss = transformer_options.get("sample_sigmas", None)
        if ss is not None and hasattr(ss, "shape") and ss.numel() > 0:
            return int((ss.to(sigma.device) - sigma).abs().argmin())
        s = float(sigma)
        if self._last_sigma is None or s > self._last_sigma:
            self._counter = 0  # new (or restarted) sampling run
        elif s < self._last_sigma:
            self._counter += 1
        self._last_sigma = s
        return self._counter

    def __call__(self, executor, x, timestep, context,
                 transformer_options={}, **kwargs):
        payload = kwargs.get("minimax_payload") or {}
        if payload.get("keyframes") or payload.get("refs"):
            raise ValueError(
                "esse_h3_loop: the Mobius loop patch is for pure text-to-AV "
                "generation. Keyframes/references anchor content to fixed "
                "positions, which rotation would smear. For image-anchored "
                "loops use the H3 Loop Bridge route instead.")

        video, audio = x[0], x[1]
        T = int(video.shape[2])
        band_v = T - 2
        if band_v < 2 * CYCLE_STEPS or band_v % CYCLE_STEPS != 0:
            raise ValueError(
                "esse_h3_loop: %d latent steps is not a rotatable H3 clip "
                "(need 17g+5 frames with g >= 2; 158 frames is a good "
                "default)." % T)
        g = band_v // CYCLE_STEPS
        Ta = int(audio.shape[-1])
        band_a = min(Ta, int(round(85.0 * g / 3.0)))
        a0 = Ta - band_a

        sigma = (timestep.flatten()[0] / 1000.0).float()
        idx = self._step_index(sigma, transformer_options)
        off_v = (idx * self.shift_steps) % band_v
        # 17/3 audio steps per video latent step; exact when off_v % 15 == 0,
        # otherwise off by at most a third of a step (~8ms) during sampling
        # only - the final latent is canonical.
        off_a = int(round(off_v * CYCLE_FRAMES / 3.0)) % band_a if band_a else 0

        if off_v == 0:
            return executor(x, timestep, context, transformer_options, **kwargs)

        def rot_v(t, sign):
            return torch.cat(
                [t[:, :, :2], torch.roll(t[:, :, 2:], sign * off_v, dims=2)], dim=2)

        def rot_a(t, sign):
            if off_a == 0 or t.shape[-1] != Ta:
                return t
            return torch.cat(
                [t[..., :a0], torch.roll(t[..., a0:], sign * off_a, dims=-1)], dim=-1)

        x_in = [rot_v(video, -1), rot_a(audio, -1)]
        # masks ride the same rotation so pinned spans keep facing their
        # content (only relevant in exotic mask+mobius combinations)
        if kwargs.get("denoise_mask") is not None and kwargs["denoise_mask"].shape[2] == T:
            kwargs = dict(kwargs)
            kwargs["denoise_mask"] = rot_v(kwargs["denoise_mask"], -1)
        if kwargs.get("audio_denoise_mask") is not None:
            kwargs = dict(kwargs)
            kwargs["audio_denoise_mask"] = rot_a(kwargs["audio_denoise_mask"], -1)

        out = executor(x_in, timestep, context, transformer_options, **kwargs)
        return [rot_v(out[0], 1), rot_a(out[1], 1)]


class EsseH3MobiusLoopPatch:
    """Make a t2va generation seamlessly loopable by construction.

    Mobius (arXiv:2502.20307): denoise with the latent rotated along time
    by a different offset each step and rotate the prediction back, so the
    wrap point is somewhere else at every step and every frame gets
    denoised with context from across the seam. No training, no anchors,
    no post-hoc closure - the clip comes out cyclic.

    H3 adaptation: offsets are multiples of 5 latent steps so content
    spans keep matching the fixed 1-4-4-4-4 RoPE lattice; the audio latent
    rotates in sync on its 40 Hz clock (exact every 3rd offset, otherwise
    within ~8ms during sampling); the 2 leading latent steps cannot join
    the cycle, so the loop is the clip minus its first 5 frames - cut them
    with H3 Mobius Loop Trim after decoding.

    Use lengths where the audio cycle is exact: 56, 107, 158, 209, 260,
    311 frames (every third point of the 17g+5 grid). Text-to-AV only; no
    first/last frames, no references. Needs empirical tuning of
    shift_frames on real renders - this is the experimental route; the
    Loop Bridge is the proven one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift_frames": ("INT", {
                    "default": 68, "min": 17, "max": 340, "step": 17,
                    "tooltip": "How far the latent rotates per sampler step, "
                               "in pixel frames (multiples of 17 = the "
                               "latent cycle). Pick something that does not "
                               "divide the loop length evenly so offsets "
                               "visit many positions: 68 against a 158-frame "
                               "clip cycles through 9 distinct seams."}),
                "exact_av": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Restrict rotation to 51-frame multiples so "
                               "audio and video offsets align EXACTLY every "
                               "step (they are otherwise within ~8ms during "
                               "sampling, which mostly matters for lipsync)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Rotate the AV latent each denoising step (Mobius) so a "
                   "text-to-AV clip comes out seamlessly loopable. Trim with "
                   "H3 Mobius Loop Trim after decoding.")

    def patch(self, model, shift_frames=68, exact_av=False):
        _ensure_h3_ready()
        shift_frames = int(shift_frames)
        if shift_frames % CYCLE_FRAMES != 0:
            raise ValueError("esse_h3_loop: shift_frames must be a multiple "
                             "of 17 (got %d)" % shift_frames)
        if exact_av and shift_frames % 51 != 0:
            raise ValueError("esse_h3_loop: exact_av needs shift_frames in "
                             "multiples of 51 (51, 102, 153...); %d is not."
                             % shift_frames)
        shift_steps = shift_frames // CYCLE_FRAMES * CYCLE_STEPS
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "esse_h3_mobius_loop",
            _MobiusRotationWrapper(shift_steps, exact_av))
        return (m,)


class EsseH3MobiusLoopTrim:
    """Cut a Mobius-sampled clip down to its cyclic part.

    The 2 leading latent steps (5 pixel frames) sit outside the rotating
    band, so they are the only non-cyclic content - and they also absorb
    the video VAE's cold start, which leaves the wrap landing on a frame
    decoded with warm temporal state. Audio is cut at the same instant and
    normalized to frames/fps so the loop's sound wraps where its picture
    does.

    wrap_crossfade_frames is a fallback for renders where rotation did not
    fully converge: it folds the last frames onto the first (shortening
    the loop by that much, sound included). A converged Mobius render
    should not need it - start at 0.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "audio": ("AUDIO",),
                "wrap_crossfade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 24,
                    "tooltip": "Fold this many trailing frames onto the "
                               "opening ones (crossfade), shortening the "
                               "loop by the same amount. 0 trusts the "
                               "rotation; use 4-8 only if a render still "
                               "shows a hitch."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                  "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "trim"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Drop the 5 non-cyclic lead frames of a Mobius-sampled "
                   "H3 clip and cut the audio to match, leaving the pure "
                   "loop.")

    LEAD_FRAMES = 5  # pixel frames covered by the 2 static latent steps

    def trim(self, images, audio=None, wrap_crossfade_frames=0, fps=24.0):
        total = int(images.shape[0])
        if not is_native_length(total):
            _LOG.warning("esse_h3_loop: %d frames is not on H3's 17g+5 grid; "
                         "trimming the standard %d lead frames anyway.",
                         total, self.LEAD_FRAMES)
        if total <= self.LEAD_FRAMES:
            raise ValueError("esse_h3_loop: clip of %d frames has nothing "
                             "left after the %d frame lead trim."
                             % (total, self.LEAD_FRAMES))
        loop = images[self.LEAD_FRAMES:]

        out_audio = None
        if audio is not None:
            sr = int(audio["sample_rate"])
            wav = audio["waveform"]
            cut = int(round(self.LEAD_FRAMES / fps * sr))
            if cut >= wav.shape[-1]:
                raise ValueError("esse_h3_loop: audio too short for the lead "
                                 "trim; check fps.")
            wav = _match_audio_len(wav[..., cut:], sr, loop.shape[0] / fps,
                                   "mobius loop")
            out_audio = {"waveform": wav, "sample_rate": sr}

        fold = int(wrap_crossfade_frames)
        fold = min(fold, loop.shape[0] // 3)
        if fold > 0:
            head = _crossfade(loop[-fold:], loop[:fold], rising=True)
            loop = torch.cat([head, loop[fold:-fold]], dim=0)
            if out_audio is not None:
                sr = out_audio["sample_rate"]
                wav = out_audio["waveform"]
                j = int(round(fold / fps * sr))
                head_w = _crossfade_wave(wav[..., -j:], wav[..., :j], rising=True)
                wav = torch.cat([head_w, wav[..., j:-j]], dim=-1)
                out_audio = {"waveform": _match_audio_len(
                    wav, sr, loop.shape[0] / fps, "mobius fold"),
                    "sample_rate": sr}

        report = ("mobius loop: %d frames (%.2fs), lead %d frames cut%s"
                  % (loop.shape[0], loop.shape[0] / fps, self.LEAD_FRAMES,
                     ", %d folded at the wrap" % fold if fold else ""))
        return (loop, out_audio, report)


# ---------------------------------------------------------------------------
# shared utility
# ---------------------------------------------------------------------------

class EsseH3LoopSeamPreview:
    """Roll a loop by half so the wrap plays mid-clip.

    A seam at the very ends of a player's timeline is invisible unless the
    player itself loops gaplessly, which most preview players do not. This
    puts the wrap in the middle of a single linear play-through, where any
    hitch, pop or colour step is obvious. Judge the loop here; ship the
    unrolled version.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                  "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "roll"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Roll a finished loop by half its length so the wrap "
                   "point plays in the middle, where a preview player can "
                   "actually show it.")

    def roll(self, images, audio=None, fps=24.0):
        n = int(images.shape[0])
        half = n // 2
        rolled = torch.cat([images[half:], images[:half]], dim=0)
        out_audio = None
        if audio is not None:
            sr = int(audio["sample_rate"])
            wav = _match_audio_len(audio["waveform"], sr, n / fps, "preview")
            cut = int(round(half / fps * sr))
            out_audio = {"waveform": torch.cat([wav[..., cut:], wav[..., :cut]],
                                               dim=-1),
                         "sample_rate": sr}
        return (rolled, out_audio)


NODE_CLASS_MAPPINGS = {
    "EsseH3LoopBridge": EsseH3LoopBridge,
    "EsseH3LoopSplice": EsseH3LoopSplice,
    "EsseH3MobiusLoopPatch": EsseH3MobiusLoopPatch,
    "EsseH3MobiusLoopTrim": EsseH3MobiusLoopTrim,
    "EsseH3LoopSeamPreview": EsseH3LoopSeamPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EsseH3LoopBridge": "H3 Loop Bridge",
    "EsseH3LoopSplice": "H3 Loop Splice",
    "EsseH3MobiusLoopPatch": "H3 Mobius Loop Patch",
    "EsseH3MobiusLoopTrim": "H3 Mobius Loop Trim",
    "EsseH3LoopSeamPreview": "H3 Loop Seam Preview",
}
