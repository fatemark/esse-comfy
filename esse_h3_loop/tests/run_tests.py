"""Self-contained tests for esse_h3_loop. No ComfyUI needed.

Stubs the four comfy surfaces the nodes touch (nested_tensor,
patcher_extension, ldm.minimax.model, node_helpers) with
behaviour-faithful minimums, then exercises the grid arithmetic, both
bridge pin modes, the splice, the Mobius rotation wrapper, and the trims
against synthetic tensors with known values.

    python3 esse_h3_loop/tests/run_tests.py
"""

import math
import os
import sys
import types

import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)


# ---------------------------------------------------------------------------
# comfy stubs (registered before importing nodes)
# ---------------------------------------------------------------------------

def _install_stubs():
    comfy = types.ModuleType("comfy")

    nt = types.ModuleType("comfy.nested_tensor")

    class NestedTensor:  # the master API surface the nodes rely on
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

        def unbind(self):
            return self.tensors

        @property
        def shape(self):
            return self.tensors[0].shape

    nt.NestedTensor = NestedTensor

    pe = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"

    pe.WrappersMP = WrappersMP

    ldm = types.ModuleType("comfy.ldm")
    minimax = types.ModuleType("comfy.ldm.minimax")
    mm = types.ModuleType("comfy.ldm.minimax.model")
    mm.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    mm.FRAME_RESCALE = 5.0 / 3.0

    class PackedLayout:  # current-master signature: no frame_count
        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None):
            pass

    mm.PackedLayout = PackedLayout

    nh = types.ModuleType("node_helpers")

    def conditioning_set_values(cond, values, append=False):
        out = []
        for emb, extra in cond:
            d = extra.copy()
            for k, v in values.items():
                d[k] = (d.get(k, []) + v) if append and k in d else v
            out.append([emb, d])
        return out

    nh.conditioning_set_values = conditioning_set_values

    comfy.nested_tensor = nt
    comfy.patcher_extension = pe
    comfy.ldm = ldm
    ldm.minimax = minimax
    minimax.model = mm
    for name, mod in [("comfy", comfy), ("comfy.nested_tensor", nt),
                      ("comfy.patcher_extension", pe), ("comfy.ldm", ldm),
                      ("comfy.ldm.minimax", minimax),
                      ("comfy.ldm.minimax.model", mm), ("node_helpers", nh)]:
        sys.modules[name] = mod
    return NestedTensor


NestedTensor = _install_stubs()

import grid  # noqa: E402
import nodes  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, (" - " + str(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def av_latent(frames, h=6, w=8, tag=0.0):
    """AV latent for a 17g+5 clip whose video steps are numbered tag+step
    and audio steps numbered tag+1000+step, so slices are identifiable."""
    t = grid.video_latent_t(frames)
    ta = grid.audio_t_for_frames(frames)
    video = torch.zeros(1, 24, t, h, w)
    video += torch.arange(t, dtype=torch.float32).view(1, 1, t, 1, 1) + tag
    audio = torch.zeros(1, 32, 2, ta)
    audio += torch.arange(ta, dtype=torch.float32).view(1, 1, 1, ta) + tag + 1000.0
    return {"samples": NestedTensor((video, audio))}


def frames_tensor(n, tag=0.0):
    return (torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1) + tag).expand(n, 4, 4, 3).contiguous()


def audio_dict(frames, sr=32000, fps=24.0, tag=0.0):
    n = int(round(frames / fps * sr))
    wav = torch.arange(n, dtype=torch.float32).view(1, 1, n) + tag
    return {"waveform": wav, "sample_rate": sr}


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_grid():
    print("grid arithmetic")
    check("pixel_frames(37) == 124", grid.pixel_frames(37) == 124)
    check("pixel_frames(2) == 5", grid.pixel_frames(2) == 5)
    check("video_latent_t inverts pixel_frames",
          all(grid.video_latent_t(grid.pixel_frames(5 * g + 2)) == 5 * g + 2
              for g in range(0, 22)))
    check("steps_for_frames windows",
          [grid.steps_for_frames(n) for n in (5, 22, 39, 56)] == [2, 7, 12, 17])
    check("steps_for_frames(21) is None", grid.steps_for_frames(21) is None)
    check("align_frame_count(120) == 124", grid.align_frame_count(120) == 124)
    check("align_frame_count(124) == 124", grid.align_frame_count(124) == 124)
    check("snap_window_down(30) == 22", grid.snap_window_down(30) == 22)
    check("audio_t 124 -> 207", grid.audio_t_for_frames(124) == 207)
    check("audio_t 158 -> 263", grid.audio_t_for_frames(158) == 263)
    check("overhang 124/207 == +1/3",
          close(grid.audio_overhang(124, 207), 1.0 / 3.0))
    check("overhang 243/405 == 0", close(grid.audio_overhang(243, 405), 0.0))
    check("step_offsets(7)", grid.step_offsets(7) == [0, 1, 5, 9, 13, 17, 18])


# ---------------------------------------------------------------------------
# bridge: conditioning_rows
# ---------------------------------------------------------------------------

def test_bridge_cond_rows():
    print("bridge (conditioning_rows)")
    src = av_latent(124, tag=0.0)      # 37 steps, audio 207
    bridge = av_latent(124, tag=500.0)
    cond = [[torch.zeros(1, 3, 8), {"some_key": 1}]]
    node = nodes.EsseH3LoopBridge()
    out_cond, out_latent, info, report = node.build(
        cond, bridge, src, "22", "22", 24, 24, "conditioning_rows")

    kfs = out_cond[0][1]["minimax_keyframes"]
    vid_kfs = [k for k in kfs if "latent" in k]
    aud_kfs = [k for k in kfs if "audio_latent" in k]
    check("14 video keyframes", len(vid_kfs) == 14, len(vid_kfs))
    check("2 audio keyframes", len(aud_kfs) == 2, len(aud_kfs))

    head_idx = [k["resolved_frame_index"] for k in vid_kfs[:7]]
    tail_idx = [k["resolved_frame_index"] for k in vid_kfs[7:]]
    check("head indices at real step offsets",
          head_idx == [0, 1, 5, 9, 13, 17, 18], head_idx)
    check("tail indices at 102 + offsets",
          tail_idx == [102, 103, 107, 111, 115, 119, 120], tail_idx)

    head_steps = [float(k["latent"][0, 0, 0, 0, 0]) for k in vid_kfs[:7]]
    tail_steps = [float(k["latent"][0, 0, 0, 0, 0]) for k in vid_kfs[7:]]
    check("head content = source steps 30..36",
          head_steps == [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0], head_steps)
    check("tail content = source steps 0..6",
          tail_steps == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tail_steps)
    check("keyframe blocks are 1-step [1,24,1,h,w]",
          all(tuple(k["latent"].shape) == (1, 24, 1, 6, 8) for k in vid_kfs))

    # audio head: 24 frames -> 40 steps, end coord snapped from 22 + 1/3
    # overhang: end_frame 22.2, index 22.2 - 24 = -1.8
    ah, at = aud_kfs
    check("audio head window is 40 steps", ah["audio_latent"].shape[-1] == 40)
    check("audio head index == -1.8",
          close(ah["resolved_frame_index"], -1.8, 1e-9),
          ah["resolved_frame_index"])
    check("audio head content = source audio 167..206",
          close(float(ah["audio_latent"][0, 0, 0, 0]), 1167.0)
          and close(float(ah["audio_latent"][0, 0, 0, -1]), 1206.0))
    # audio tail: clamped to 22 frames -> round(36.67) = 37 steps at index 102
    check("audio tail index == 102", at["resolved_frame_index"] == 102)
    check("audio tail is 37 steps of source audio head",
          at["audio_latent"].shape[-1] == 37
          and close(float(at["audio_latent"][0, 0, 0, 0]), 1000.0))

    check("original conditioning untouched",
          "minimax_keyframes" not in cond[0][1] and out_cond[0][1]["some_key"] == 1)
    check("latent passed through in cond mode", out_latent is bridge)
    check("info spans", (info["head_span"], info["tail_span"],
                         info["bridge_frames"], info["source_frames"])
          == (22, 22, 124, 124))

    # upstream anchors: one in a pinned span (dropped), one mid (kept)
    cond2 = [[torch.zeros(1, 3, 8), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 6, 8)},
        {"resolved_frame_index": 60, "latent": torch.zeros(1, 24, 1, 6, 8)},
        {"resolved_frame_index": 123, "latent": torch.zeros(1, 24, 1, 6, 8)},
    ]}]]
    out2, _, _, _ = node.build(cond2, bridge, src, "22", "22", 24, 24,
                               "conditioning_rows")
    kept = [k["resolved_frame_index"] for k in out2[0][1]["minimax_keyframes"]
            if "latent" in k]
    check("anchor at 60 kept, 123 dropped, 15 video kfs total",
          60 in kept and 123 not in kept and len(kept) == 15, kept)

    # too-small middle refused
    small = av_latent(56, tag=500.0)
    try:
        node.build(cond, small, src, "22", "22", 24, 24, "conditioning_rows")
        check("tiny bridge refused", False)
    except ValueError:
        check("tiny bridge refused", True)


# ---------------------------------------------------------------------------
# bridge: latent_mask
# ---------------------------------------------------------------------------

def test_bridge_mask():
    print("bridge (latent_mask)")
    src = av_latent(124, tag=0.0)
    bridge = av_latent(124, tag=500.0)
    cond = [[torch.zeros(1, 3, 8), {}]]
    node = nodes.EsseH3LoopBridge()
    out_cond, out_latent, info, report = node.build(
        cond, bridge, src, "22", "22", 24, 24, "latent_mask")

    video, audio = out_latent["samples"].unbind()
    vmask, amask = out_latent["noise_mask"].unbind()
    check("video head = source tail steps",
          close(float(video[0, 0, 0, 0, 0]), 30.0)
          and close(float(video[0, 0, 6, 0, 0]), 36.0))
    check("video tail = source head steps",
          close(float(video[0, 0, 30, 0, 0]), 0.0)
          and close(float(video[0, 0, 36, 0, 0]), 6.0))
    check("video middle stays empty", float(video[0, 0, 15].abs().sum()) == 0.0)
    check("vmask zero exactly on pinned steps",
          float(vmask[0, 0, :7].sum()) == 0.0
          and float(vmask[0, 0, 30:].sum()) == 0.0
          and bool((vmask[0, 0, 7:30] == 1.0).all()))
    check("vmask full latent shape", tuple(vmask.shape) == (1, 24, 37, 6, 8))

    # audio: head 40 steps pinned at [0:40]; tail slot round(102*5/3)=170,
    # 37 steps -> [170:207], mask zero from 170
    check("audio head pinned [0:40]",
          close(float(audio[0, 0, 0, 0]), 1167.0)
          and float(amask[0, 0, 0, :40].sum()) == 0.0)
    check("audio tail pinned from slot 170",
          close(float(audio[0, 0, 0, 170]), 1000.0)
          and float(amask[0, 0, 0, 170:].sum()) == 0.0
          and bool((amask[0, 0, 0, 40:170] == 1.0).all()))
    check("no keyframes added in mask mode",
          "minimax_keyframes" not in out_cond[0][1])


# ---------------------------------------------------------------------------
# splice
# ---------------------------------------------------------------------------

def test_splice():
    print("splice")
    info = {"kind": "bridge", "pin_mode": "conditioning_rows",
            "head_span": 22, "tail_span": 22, "bridge_frames": 124,
            "source_frames": 124, "fps": 24}
    a = frames_tensor(124, 0.0)
    b = frames_tensor(124, 1000.0)
    node = nodes.EsseH3LoopSplice()
    loop, audio, report = node.splice(a, b, info,
                                      audio_a=audio_dict(124, tag=0.0),
                                      audio_bridge=audio_dict(124, tag=50000.0),
                                      video_blend_frames=0,
                                      audio_crossfade_ms=0,
                                      seam_inset_frames=0)
    check("loop length 204", loop.shape[0] == 204, loop.shape[0])
    check("A kept verbatim", float(loop[0, 0, 0, 0]) == 0.0
          and float(loop[123, 0, 0, 0]) == 123.0)
    check("middle = bridge frames 22..101",
          float(loop[124, 0, 0, 0]) == 1022.0
          and float(loop[203, 0, 0, 0]) == 1101.0)
    sr = audio["sample_rate"]
    check("audio length exact 204/24 s",
          audio["waveform"].shape[-1] == int(round(204 / 24 * sr)),
          audio["waveform"].shape[-1])
    # audio middle starts at the bridge's 22/24s sample
    a_len = int(round(124 / 24 * sr))
    first_mid = float(audio["waveform"][0, 0, a_len])
    check("audio middle from bridge at 22/24s",
          close(first_mid, 50000.0 + round(22 / 24 * sr)), first_mid)

    # blends stay in range and keep length
    loop2, audio2, _ = node.splice(a, b, info,
                                   audio_a=audio_dict(124, tag=0.0),
                                   audio_bridge=audio_dict(124, tag=50000.0),
                                   video_blend_frames=6,
                                   audio_crossfade_ms=20,
                                   seam_inset_frames=0)
    check("blended loop same length", loop2.shape[0] == 204
          and audio2["waveform"].shape[-1] == audio["waveform"].shape[-1])
    check("wrap blend mixes bridge material into A's opening",
          float(loop2[0, 0, 0, 0]) != 0.0 and float(loop2[10, 0, 0, 0]) == 10.0)

    # seam inset (default 5): cuts move inside the pinned spans, so each
    # join sits between a real A frame and its pinned counterpart
    loop3, audio3, report3 = node.splice(
        a, b, info, audio_a=audio_dict(124, tag=0.0),
        audio_bridge=audio_dict(124, tag=50000.0),
        video_blend_frames=0, audio_crossfade_ms=0)
    check("inset loop still 204 frames", loop3.shape[0] == 204,
          loop3.shape[0])
    check("inset loop starts at A frame 5", float(loop3[0, 0, 0, 0]) == 5.0)
    check("A part ends at true frame 118 then bridge copy of 119",
          float(loop3[113, 0, 0, 0]) == 118.0
          and float(loop3[114, 0, 0, 0]) == 1017.0)
    check("inset loop ends on bridge copy of A frame 4",
          float(loop3[203, 0, 0, 0]) == 1106.0)
    sr3 = audio3["sample_rate"]
    check("inset audio starts at the 5/24s sample of A",
          close(float(audio3["waveform"][0, 0, 0]), round(5 / 24 * sr3)))
    check("inset audio length exact",
          audio3["waveform"].shape[-1] == int(round(204 / 24 * sr3)))
    check("report carries seam metrics",
          "seam deltas" in report3 and "inset 5" in report3)

    # frame-count mismatch on the bridge is refused
    try:
        node.splice(a, frames_tensor(107, 1000.0), info)
        check("bridge frame mismatch refused", False)
    except ValueError:
        check("bridge frame mismatch refused", True)


# ---------------------------------------------------------------------------
# mobius wrapper
# ---------------------------------------------------------------------------

def test_mobius():
    print("mobius rotation wrapper")
    W = nodes._MobiusRotationWrapper(20, False)  # 68 frames -> 20 steps
    T, Ta = 47, 263                              # 158-frame clip
    video = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1, 1).expand(1, 24, T, 6, 8).contiguous()
    audio = torch.arange(Ta, dtype=torch.float32).view(1, 1, 1, Ta).expand(1, 32, 2, Ta).contiguous()
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    seen = {}

    def executor(x, timestep, context, transformer_options, **kw):
        seen["video"], seen["audio"] = x[0], x[1]
        return [x[0].clone(), x[1].clone()]

    def call(sigma):
        ts = torch.tensor([sigma * 1000.0])
        return W(executor, [video, audio], ts, torch.zeros(1, 8, 16),
                 {"sample_sigmas": sigmas}, minimax_payload={})

    out = call(1.0)  # idx 0 -> offset 0: passthrough
    check("offset 0 passthrough", torch.equal(out[0], video) and torch.equal(out[1], audio))

    out = call(0.8)  # idx 1 -> off_v 20, off_a round(20*17/3)=113
    v_in = seen["video"]
    check("static head untouched",
          torch.equal(v_in[:, :, :2], video[:, :, :2]))
    check("band rolled left by 20",
          float(v_in[0, 0, 2, 0, 0]) == 22.0
          and float(v_in[0, 0, 46, 0, 0]) == float((44 + 20) % 45 + 2))
    a_in = seen["audio"]
    a0 = Ta - 255
    check("audio static head untouched",
          torch.equal(a_in[..., :a0], audio[..., :a0]))
    check("audio band rolled by 113",
          float(a_in[0, 0, 0, a0]) == float((113) % 255 + a0))
    check("output un-rolled to canonical",
          torch.equal(out[0], video) and torch.equal(out[1], audio))

    # same sigma again (CFG's second call) -> same offset
    call(0.8)
    check("same sigma -> same rotation",
          float(seen["video"][0, 0, 2, 0, 0]) == 22.0)

    # offsets stay multiples of 5 and cycle through many positions
    offs = set()
    for i, s in enumerate(sigmas[:-1]):
        call(float(s))
        off = (int(float(seen["video"][0, 0, 2, 0, 0])) - 2) % 45
        offs.add(off)
        check("offset %d is a cycle multiple" % off, off % 5 == 0)
    check("multiple distinct seam positions", len(offs) >= 4, offs)

    # keyframes refused
    try:
        ts = torch.tensor([800.0])
        W(executor, [video, audio], ts, torch.zeros(1, 8, 16),
          {"sample_sigmas": sigmas},
          minimax_payload={"keyframes": [{"resolved_frame_index": 0}]})
        check("keyframes refused", False)
    except ValueError:
        check("keyframes refused", True)

    # fallback step counter when sample_sigmas is absent
    W2 = nodes._MobiusRotationWrapper(20, False)
    for expect, s in [(0, 1.0), (0, 1.0), (1, 0.8), (1, 0.8), (2, 0.6)]:
        ts = torch.tensor([s * 1000.0])
        W2(executor, [video, audio], ts, torch.zeros(1, 8, 16), {},
           minimax_payload={})
        got = (int(float(seen["video"][0, 0, 2, 0, 0])) - 2) % 45
        check("fallback idx %d at sigma %.1f" % (expect, s),
              got == (expect * 20) % 45, got)

    # patch node validation
    try:
        nodes.EsseH3MobiusLoopPatch().patch(None, shift_frames=20)
        check("off-grid shift refused", False)
    except ValueError:
        check("off-grid shift refused", True)
    try:
        nodes.EsseH3MobiusLoopPatch().patch(None, shift_frames=68, exact_av=True)
        check("exact_av off 51-grid refused", False)
    except ValueError:
        check("exact_av off 51-grid refused", True)


# ---------------------------------------------------------------------------
# mobius trim + seam preview
# ---------------------------------------------------------------------------

def test_trim_and_preview():
    print("mobius trim / seam preview")
    node = nodes.EsseH3MobiusLoopTrim()
    imgs = frames_tensor(158)
    loop, audio, report = node.trim(imgs, audio=audio_dict(158), fps=24.0)
    check("lead 5 frames cut", loop.shape[0] == 153
          and float(loop[0, 0, 0, 0]) == 5.0)
    sr = audio["sample_rate"]
    check("audio cut to 153/24 s",
          audio["waveform"].shape[-1] == int(round(153 / 24 * sr)))
    check("audio starts at the 5/24s sample",
          close(float(audio["waveform"][0, 0, 0]), round(5 / 24 * sr)))

    loop2, audio2, _ = node.trim(imgs, audio=audio_dict(158), fps=24.0,
                                 wrap_crossfade_frames=6)
    check("fold shortens by 6", loop2.shape[0] == 147
          and audio2["waveform"].shape[-1] == int(round(147 / 24 * sr)))

    prev = nodes.EsseH3LoopSeamPreview()
    rolled, raudio = prev.roll(frames_tensor(100), audio=audio_dict(100), fps=24.0)
    check("preview rolls by half", float(rolled[0, 0, 0, 0]) == 50.0
          and float(rolled[50, 0, 0, 0]) == 0.0
          and rolled.shape[0] == 100)
    check("preview audio rolled in sync",
          close(float(raudio["waveform"][0, 0, 0]), round(50 / 24 * 32000)))


def test_old_layout_gating():
    """First-generation H3 layout (PackedLayout with frame_count, e.g.
    ComfyUI v0.30.x): conditioning_rows must refuse with advice, while
    latent_mask and the Mobius patch keep working."""
    print("old-layout gating")
    mm = sys.modules["comfy.ldm.minimax.model"]
    new_layout = mm.PackedLayout

    class OldPackedLayout:
        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
            pass

    mm.PackedLayout = OldPackedLayout
    nodes._h3_state = None
    try:
        caps = nodes._h3_capabilities()
        check("old layout detected", caps["arbitrary_anchors"] is False)

        src = av_latent(124, tag=0.0)
        bridge = av_latent(124, tag=500.0)
        cond = [[torch.zeros(1, 3, 8), {}]]
        node = nodes.EsseH3LoopBridge()
        try:
            node.build(cond, bridge, src, "22", "22", 24, 24,
                       "conditioning_rows")
            check("conditioning_rows refused on old layout", False)
        except ValueError as exc:
            check("conditioning_rows refused on old layout",
                  "latent_mask" in str(exc))

        _, out_latent, _, _ = node.build(cond, bridge, src, "22", "22",
                                         24, 24, "latent_mask")
        check("latent_mask still works on old layout",
              "noise_mask" in out_latent)

        try:
            nodes.EsseH3MobiusLoopTrim()  # trivially constructible
            nodes._MobiusRotationWrapper(20, False)
            nodes._ensure_h3_ready()
            check("mobius path not blocked by old layout", True)
        except Exception as exc:
            check("mobius path not blocked by old layout", False, exc)
    finally:
        mm.PackedLayout = new_layout
        nodes._h3_state = None


if __name__ == "__main__":
    test_grid()
    test_bridge_cond_rows()
    test_bridge_mask()
    test_splice()
    test_mobius()
    test_trim_and_preview()
    test_old_layout_gating()
    print()
    if FAILS:
        print("%d FAILURES: %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("all checks passed")
