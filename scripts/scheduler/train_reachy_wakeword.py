#!/usr/bin/env python3
"""Train a custom openWakeWord model for the phrase "Reachy Wake up".

RUN THIS OFF-DEVICE (Google Colab, or any x86_64 Linux box with a few GB free and
internet) -- NOT on the Raspberry Pi. It generates synthetic speech, downloads
negative/background feature sets, and trains a small classifier on top of
openWakeWord's shared embedding model, exporting a single `.onnx` file.

Then copy the resulting model to the robot and point the service at it:

    # on your training machine, after this finishes:
    scp ./reachy_wake_up/reachy_wake_up.onnx \
        pollen@<robot-ip>:/home/pollen/clawbody/scripts/scheduler/models/

    # on the robot:
    sudo sed -i 's#^Environment=REACHY_WAKE_MODEL=.*#Environment=REACHY_WAKE_MODEL=/home/pollen/clawbody/scripts/scheduler/models/reachy_wake_up.onnx#' \
        /etc/systemd/system/reachy-wake.service
    sudo systemctl daemon-reload && sudo systemctl restart reachy-wake.service
    journalctl -u reachy-wake.service -n 20 --no-pager   # expect: loaded wakeword(s): ['reachy_wake_up']

----------------------------------------------------------------------------
Colab quickstart (paste into a GPU runtime cell):

    !git clone https://github.com/dscripka/openWakeWord /content/openWakeWord
    !wget -O /content/train_reachy_wakeword.py <raw-url-of-this-file>
    !python /content/train_reachy_wakeword.py --work-dir /content

This mirrors openWakeWord's official `notebooks/automatic_model_training.ipynb`
(https://github.com/dscripka/openWakeWord) with the phrase preset to
"Reachy Wake up". See that notebook if any upstream API has drifted.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

TARGET_PHRASE = "Reachy Wake up"
MODEL_NAME = "reachy_wake_up"

# Tune these up for a stronger model (more samples = better, but slower).
N_SAMPLES = 30000           # synthetic positive clips of the phrase
N_SAMPLES_VAL = 2000
TRAIN_STEPS = 50000
TARGET_RECALL = 0.5
MAX_NEGATIVE_WEIGHT = 1500


def sh(cmd: list[str], cwd: str | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def pip_install(*pkgs: str) -> None:
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def write_config(work: Path, oww_repo: Path) -> Path:
    """Write the training YAML consumed by openWakeWord's train.py."""
    out_dir = work / MODEL_NAME
    cfg = textwrap.dedent(f"""\
        # Auto-generated config for the "{TARGET_PHRASE}" wake word.
        target_phrase:
          - "{TARGET_PHRASE.lower()}"
        model_name: "{MODEL_NAME}"

        n_samples: {N_SAMPLES}
        n_samples_val: {N_SAMPLES_VAL}
        steps: {TRAIN_STEPS}
        max_negative_weight: {MAX_NEGATIVE_WEIGHT}
        target_accuracy: 0.7
        target_recall: {TARGET_RECALL}

        output_dir: "{out_dir}"
        # Room-impulse-responses + background audio for augmentation (downloaded below)
        rir_paths:
          - "{work / 'mit_rirs'}"
        background_paths:
          - "{work / 'audioset_16k'}"
          - "{work / 'fma'}"
        # Precomputed negative features + a false-positive validation set
        false_positive_validation_data_path: "{work / 'validation_set_features.npy'}"
        feature_data_files:
          ACAV100M_sample: "{work / 'openwakeword_features_ACAV100M_2000_hrs_16bit.npy'}"

        # Synthetic-speech generation (piper-sample-generator)
        tts_batch_size: 50
        augmentation_batch_size: 16
        augmentation_rounds: 1
        piper_sample_generator_path: "{work / 'piper-sample-generator'}"
    """)
    cfg_path = work / f"{MODEL_NAME}.yaml"
    cfg_path.write_text(cfg)
    print(f"wrote training config -> {cfg_path}")
    return cfg_path


def fetch_assets(work: Path, oww_repo: Path) -> None:
    """Clone piper-sample-generator and download the datasets the config expects.

    openWakeWord ships a helper that downloads MIT RIRs, AudioSet/FMA backgrounds,
    the ACAV100M negative features, and a validation set. We reuse it so the paths
    in the YAML resolve. (This is the same data the official notebook pulls.)
    """
    if not (work / "piper-sample-generator").exists():
        sh(["git", "clone", "https://github.com/rhasspy/piper-sample-generator",
            str(work / "piper-sample-generator")])
        # Default English voice used by the openWakeWord notebook.
        sh(["wget", "-O", str(work / "piper-sample-generator" / "models" / "en_US-libritts_r-medium.pt"),
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"])

    # openWakeWord bundles a data-download utility used by the training notebook.
    sh([sys.executable, "-c", textwrap.dedent(f"""
        import os
        os.chdir(r"{work}")
        from openwakeword.data import download_training_data
        download_training_data()  # MIT RIRs, audioset_16k, fma, ACAV100M features, validation set
    """)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the 'Reachy Wake up' openWakeWord model.")
    ap.add_argument("--work-dir", default=".", help="Working dir for clones/datasets/output")
    ap.add_argument("--oww-repo", default=None, help="Path to a cloned openWakeWord repo (else cloned into work-dir)")
    ap.add_argument("--skip-assets", action="store_true", help="Skip dataset download (already present)")
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    oww_repo = Path(args.oww_repo).resolve() if args.oww_repo else (work / "openWakeWord")
    if not oww_repo.exists():
        sh(["git", "clone", "https://github.com/dscripka/openWakeWord", str(oww_repo)])

    print("installing training dependencies (torch, openwakeword[full], etc.)...")
    pip_install("openwakeword", "torch", "torchaudio", "torchinfo", "torchmetrics",
                "speechbrain", "audiomentations", "torch-audiomentations",
                "acoustics", "scipy", "numpy", "tqdm", "mutagen", "pyyaml")

    if not args.skip_assets:
        fetch_assets(work, oww_repo)

    cfg = write_config(work, oww_repo)
    train_py = oww_repo / "openwakeword" / "train.py"

    # Three official stages: generate synthetic clips, augment them, train the model.
    sh([sys.executable, str(train_py), "--training_config", str(cfg), "--generate_clips"], cwd=str(work))
    sh([sys.executable, str(train_py), "--training_config", str(cfg), "--augment_clips"], cwd=str(work))
    sh([sys.executable, str(train_py), "--training_config", str(cfg), "--train_model"], cwd=str(work))

    onnx = work / MODEL_NAME / f"{MODEL_NAME}.onnx"
    print("\n" + "=" * 60)
    if onnx.is_file():
        print(f"DONE. Trained model: {onnx}")
        print("Copy it to the robot's scripts/scheduler/models/ and repoint the service "
              "(see the deploy notes at the top of this file).")
    else:
        print("Training finished but the .onnx was not found at the expected path.")
        print(f"Look under: {work / MODEL_NAME}")
        sys.exit(1)


if __name__ == "__main__":
    main()
