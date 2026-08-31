"""esse_h3_loop - perfect AV loops for MiniMax H3 in ComfyUI.

Two routes:
  - Loop Bridge: close an existing clip into a loop by pinning a bridge's
    head to the clip's ending and its tail to the clip's opening (latent
    handoff, both picture and sound), then splicing the free middle.
  - Mobius: rotate the AV latent during denoising so a t2va clip comes out
    cyclic by construction (arXiv:2502.20307, adapted to H3's grids).

See README.md and example_workflows/.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
