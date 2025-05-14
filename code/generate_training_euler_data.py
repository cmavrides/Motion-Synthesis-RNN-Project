
"""
Unified BVH ⇄ Euler-angle training-data converter with optional reconstruction.
"""

from pathlib import Path
import math
import numpy as np
import read_bvh

# scale factor: metres → tens of cm
TRANSLATION_SCALE = 0.01


def generate_euler_traindata_from_bvh(
    src_bvh_folder: str,
    tar_traindata_folder: str
) -> None:
    """
    Encode  *.bvh of one dance as a normalized NumPy array (F×C):
      • root translation [m] → scaled by TRANSLATION_SCALE
      • Euler angles [deg] → radians
    Saves each as *.npy in `train_data_euler`.
    """
    process_motion_data(
        source=src_bvh_folder,
        target=tar_traindata_folder,
        mode="encode"
    )


def generate_bvh_from_euler_traindata(
    src_train_folder: str,
    tar_bvh_folder: str,
    template_bvh: str | None = None
) -> None:
    """
    Decode each *.npy in `src_train_folder` back to BVH format:
      • denormalize translation back to metres
      • Euler angles [rad] → degrees
    Uses `template_bvh` for hierarchy (default standard.bvh).
    """
    process_motion_data(
        source=src_train_folder,
        target=tar_bvh_folder,
        mode="decode",
        template=template_bvh
    )


def process_motion_data(
    source: str,
    target: str,
    mode: str = "encode",
    template: str | None = None
) -> None:
    """
    Core converter: BVH ↔ NumPy .npy files.

    Args:
        source: directory of .bvh files (encode) or .npy files (decode).
        target: output directory.
        mode: 'encode' to BVH→.npy, 'decode' to .npy→BVH.
        template: BVH hierarchy template for decode.
    """
    src_dir = Path(source)
    tgt_dir = Path(target)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    if mode == "decode":
        if template is None:
            template = (
                Path(__file__).parent.parent
                / "train_data_bvh"
                / "standard.bvh"
            )
        template = Path(template)

    for entry in src_dir.iterdir():
        if mode == "encode" and entry.suffix.lower() == ".bvh":
            frames = read_bvh.parse_frames(str(entry)).astype(np.float32)
            frames[:, :3] *= TRANSLATION_SCALE
            frames[:, 3:] *= math.pi / 180.0
            out_f = tgt_dir / f"{entry.stem}.npy"
            np.save(str(out_f), frames)
            print(f"Saved encoded data: {out_f}")

        elif mode == "decode" and entry.suffix.lower() == ".npy":
            frames = np.load(str(entry)).astype(np.float32)
            frames[:, :3] /= TRANSLATION_SCALE
            frames[:, 3:] *= 180.0 / math.pi
            out_f = tgt_dir / f"{entry.stem}.bvh"
            read_bvh.write_frames(str(template), str(out_f), frames)
            print(f"Reconstructed BVH: {out_f}")

        # skip other file types automatically

    if mode not in ("encode", "decode"):
        raise ValueError(f"Invalid mode '{mode}'. Use 'encode' or 'decode'.")


if __name__ == "__main__":
    # ===== USER CONFIGURATION =====
    SRC_FOLDER   = "../train_data_bvh/martial"      # directory of .bvh or .npy
    DST_FOLDER   = "../train_data_euler/martial"   # where to write outputs
    MODE         = "encode"             # 'encode' or 'decode'
    TEMPLATE_BVH = "../train_data_bvh/standard.bvh"                  # bvH template if decoding

    # Primary conversion
    if MODE == "decode":
        generate_bvh_from_euler_traindata(
            SRC_FOLDER,
            DST_FOLDER,
            TEMPLATE_BVH
        )
        # optional: re-encode reconstructed BVH back to .npy
        recon_encode_dir = Path(DST_FOLDER + "_reencoded")
        generate_euler_traindata_from_bvh(
            str(recon_encode_dir.parent),
            str(recon_encode_dir)
        )
    else:
        generate_euler_traindata_from_bvh(
            SRC_FOLDER,
            DST_FOLDER
        )
        # optional: reconstruct BVH from encoded .npy
        recon_bvh_dir = Path(DST_FOLDER + "_reconstructed_bvh")
        generate_bvh_from_euler_traindata(
            DST_FOLDER,
            str(recon_bvh_dir),
            TEMPLATE_BVH
        )
