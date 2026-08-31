"""MiniMax H3 latent-grid arithmetic. Pure Python, no ComfyUI imports.

H3 runs three clocks that only occasionally agree, and every seam bug in a
loop comes from mixing them up:

  pixel frames   24 fps. All node-facing frame counts are in this clock.
  video latents  one latent step covers 1, 4, 4, 4, 4 pixel frames, cycling
                 with period 5 steps = 17 frames. A clip is 17g+5 frames,
                 which is 5g+2 steps, and the cycle always starts at
                 position 0 - so a run of steps only has the same internal
                 frame spans as a freshly encoded run when it starts at a
                 step index that is a multiple of 5.
  audio latents  40 Hz, so 5/3 audio steps per pixel frame. 17 frames is
                 28.33 audio steps: the two latent clocks only meet every
                 3 pixel frames (5 audio steps), and every 51 frames
                 (15 video steps = 85 audio steps) on the video-step grid.

Everything here is integer/float arithmetic over those three clocks, kept
separate from nodes.py so it can be unit-tested without ComfyUI.
"""

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
CYCLE_STEPS = 5           # video latent steps per cycle
CYCLE_FRAMES = 17         # pixel frames per cycle
FPS = 24
AUDIO_HZ = 40.0
FRAME_RESCALE = AUDIO_HZ / FPS   # 5/3 audio latent steps per pixel frame

# Pinnable context windows: the only pixel-frame runs that are a whole number
# of latent steps starting at cycle position 0 (see steps_for_frames).
CONTEXT_WINDOWS = (5, 22, 39, 56)


def pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps (from cycle position 0)."""
    latent_t = int(latent_t)
    full, rem = divmod(latent_t, CYCLE_STEPS)
    return full * CYCLE_FRAMES + sum(FRAME_PER_TOKEN[:rem])


def step_offsets(latent_t):
    """Pixel-frame index at which each of latent_t latent steps begins."""
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % CYCLE_STEPS]
    return out


def steps_for_frames(n):
    """Latent steps covering exactly n pixel frames from cycle position 0,
    or None when no whole number of steps lands on n (5 -> 2, 22 -> 7,
    39 -> 12, 56 -> 17; 1 also works but a clip tail never spans 1 frame)."""
    k, covered = 0, 0
    n = int(n)
    while covered < n:
        covered += FRAME_PER_TOKEN[k % CYCLE_STEPS]
        k += 1
    return k if covered == n else None


def align_frame_count(n):
    """Snap a frame count UP to H3's 17g+5 grid (what the stock nodes do)."""
    n = max(5, int(n))
    while n % CYCLE_FRAMES != 5:
        n += 1
    return n


def is_native_length(n):
    return int(n) >= 5 and int(n) % CYCLE_FRAMES == 5


def video_latent_t(frame_count):
    """Latent steps for a 17g+5 pixel-frame clip (the VAE's downscale formula)."""
    frame_count = int(frame_count)
    return 2 if frame_count <= 5 else ((frame_count - 5) // CYCLE_FRAMES) * CYCLE_STEPS + 2


def snap_window_down(n):
    """Largest usable context window (17m+5 frames) that is <= n, or None."""
    n = int(n)
    if n < 5:
        return None
    return n - ((n - 5) % CYCLE_FRAMES)


def audio_t_for_frames(frame_count):
    """Audio latent steps H3 allocates for a clip: rounded to the NEAREST
    step, so a third of legal lengths carry ~8ms more sound than picture
    and another third ~8ms less."""
    return int(round(int(frame_count) / FPS * AUDIO_HZ))


def audio_overhang(frame_count, audio_t):
    """Signed fraction of an audio step by which the clip's audio grid
    overshoots its last pixel frame. Legal values are 0 and +-1/3; anything
    outside (-0.5, 0.5) means the latent is not on H3's grid."""
    return float(audio_t) - FRAME_RESCALE * int(frame_count)


def audio_steps_for_offset(video_steps):
    """Audio latent steps spanned by a run of video latent steps that is a
    multiple of the 5-step cycle. Exact only when video_steps is a multiple
    of 15 (51 frames = 85 audio steps); otherwise the true value has a
    1/3-step fraction and the caller must round."""
    return video_steps * CYCLE_FRAMES / 3.0  # = video_steps/5 * 17 * 5/3
