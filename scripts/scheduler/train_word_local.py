#!/usr/bin/env python3
"""Local wakeword training pipeline converted from train_word.ipynb.

This script removes Colab assumptions (/content, notebook magics, apt cells) and
runs the same flow as a plain Python program on Linux:
1) Clone/update repos
2) Download required feature assets
3) Collect AudioSet background audio
4) Generate + augment synthetic phrase speech
5) Write openWakeWord YAML config
6) Run generate_clips / augment_clips / train_model
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import datasets
import soundfile as sf
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

MODEL_NAME = "reachy_wake_up"
PHRASE_VARIANTS = [
    "reachy wake up",
    "ree-chee wake up",
    "reechee wake up",
    "reachi wake up",
]


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        input=input_text,
        text=True,
        check=True,
    )


def clone_clean(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    run(["git", "clone", "--depth", "1", repo_url, str(target_dir)])


def ensure_openwakeword_resources(openwakeword_dir: Path) -> None:
    if str(openwakeword_dir) not in sys.path:
        sys.path.insert(0, str(openwakeword_dir))

    import openwakeword
    from openwakeword.utils import download_file

    resources_models_dir = openwakeword_dir / "openwakeword" / "resources" / "models"
    resources_models_dir.mkdir(parents=True, exist_ok=True)

    feature_urls = [spec["download_url"] for spec in openwakeword.FEATURE_MODELS.values()]
    for url in feature_urls:
        tflite_name = url.rsplit("/", maxsplit=1)[-1]
        onnx_name = tflite_name.replace(".tflite", ".onnx")
        if not (resources_models_dir / tflite_name).exists():
            print(f"Downloading missing OpenWakeWord model: {tflite_name}")
            download_file(url, str(resources_models_dir))
        if not (resources_models_dir / onnx_name).exists():
            print(f"Downloading missing OpenWakeWord model: {onnx_name}")
            download_file(url.replace(".tflite", ".onnx"), str(resources_models_dir))


def download_feature_files(work: Path) -> None:
    dataset_repo = "davidscripka/openwakeword_features"
    hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename="openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        local_dir=str(work),
    )
    hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename="validation_set_features.npy",
        local_dir=str(work),
    )


def collect_audioset_background(work: Path, n_clips: int) -> None:
    out_a = work / "audioset_16k"
    out_a.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {n_clips} AudioSet clips to {out_a}...")

    ds_a = datasets.load_dataset("agkphysics/AudioSet", split="train", streaming=True)
    ds_a = iter(ds_a.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    for i in tqdm(range(n_clips), desc="AudioSet"):
        try:
            row = next(ds_a)
            sf.write(out_a / f"audioset_{i:04d}.wav", row["audio"]["array"], 16000)
        except Exception as exc:  # best-effort fetch
            print(f"Skipping clip {i}: {exc}")


def prepare_piper_sample_generator(work: Path) -> Path:
    piper_repo = work / "piper-sample-generator"
    clone_clean("https://github.com/dscripka/piper-sample-generator.git", piper_repo)

    model_path = piper_repo / "models" / "en-us-libritts-high.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "wget",
            "-q",
            "-O",
            str(model_path),
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt",
        ]
    )
    return piper_repo


def generate_raw_synthetic(work: Path, piper_model_dir: Path) -> Path:
    out_dir = work / "raw_synthetic_speech"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_wav in out_dir.glob("*.wav"):
        old_wav.unlink()

    api = HfApi()
    repo_files = api.list_repo_files("rhasspy/piper-voices")
    preferred = [
        "en_US-amy-medium.onnx",
        "en_US-ryan-high.onnx",
        "en_US-lessac-medium.onnx",
        "en_US-libritts-high.onnx",
    ]

    voices: list[tuple[Path, Path]] = []
    english_pairs: list[tuple[str, str]] = []
    for item in repo_files:
        if not (item.startswith("en/") and item.endswith(".onnx")):
            continue
        config_item = item.replace(".onnx", ".onnx.json")
        if config_item in repo_files:
            english_pairs.append((item, config_item))

    selected: list[tuple[str, str]] = []
    for target in preferred:
        for model_rel, config_rel in english_pairs:
            if Path(model_rel).name == target:
                selected.append((model_rel, config_rel))
                break
    for model_rel, config_rel in english_pairs:
        if len(selected) >= 4:
            break
        if (model_rel, config_rel) not in selected:
            selected.append((model_rel, config_rel))

    if not selected:
        raise RuntimeError("No compatible English Piper voices found.")

    for model_rel, config_rel in selected:
        model_local = piper_model_dir / model_rel
        config_local = piper_model_dir / config_rel
        model_local.parent.mkdir(parents=True, exist_ok=True)
        config_local.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=model_rel,
            local_dir=str(piper_model_dir),
        )
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=config_rel,
            local_dir=str(piper_model_dir),
        )
        voices.append((model_local, config_local))

    sample_idx = 0
    for model_path, config_path in voices:
        print(f"Generating samples with voice {model_path.name}...")
        for _ in range(8):
            phrase = random.choice(PHRASE_VARIANTS)
            output = out_dir / f"raw_sample_{sample_idx:03d}.wav"
            sample_idx += 1
            run(
                [
                    "piper",
                    "--model",
                    str(model_path),
                    "--config",
                    str(config_path),
                    "--output_file",
                    str(output),
                ],
                input_text=f"{phrase}\n",
            )

    if not any(out_dir.glob("*.wav")):
        raise RuntimeError("No raw synthetic clips were generated.")
    return out_dir


def augment_synthetic(raw_dir: Path, work: Path) -> Path:
    out_dir = work / "synthetic_speech"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "piper_sample_generator.augment",
            str(raw_dir),
            str(out_dir),
            "--sample-rate",
            "16000",
        ]
    )
    if not any(out_dir.glob("*.wav")):
        raise RuntimeError("No augmented synthetic clips were generated.")
    return out_dir


def write_config(work: Path, piper_repo: Path) -> Path:
    config_path = work / f"{MODEL_NAME}.yaml"
    out_a = work / "audioset_16k"
    out_f = work / "fma"
    out_f.mkdir(parents=True, exist_ok=True)

    background_paths = [str(out_a)]
    if any(out_f.glob("*.wav")):
        background_paths.append(str(out_f))

    config_data = {
        "target_phrase": PHRASE_VARIANTS,
        "model_name": MODEL_NAME,
        "custom_negative_phrases": [],
        "n_samples": 30000,
        "n_samples_val": 2000,
        "tts_batch_size": 50,
        "output_dir": str(work / MODEL_NAME),
        "background_paths": background_paths,
        "background_paths_duplication_rate": [1] * len(background_paths),
        "rir_paths": [],
        "false_positive_validation_data_path": str(work / "validation_set_features.npy"),
        "feature_data_files": {
            "ACAV100M_sample": str(work / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
        },
        "batch_n_per_class": {
            "ACAV100M_sample": 1024,
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": 50000,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "target_accuracy": 0.7,
        "target_recall": 0.5,
        "piper_sample_generator_path": str(piper_repo),
        "augmentation_batch_size": 16,
        "augmentation_rounds": 1,
    }

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
    return config_path


def run_training_stages(openwakeword_dir: Path, config_path: Path) -> None:
    train_dir = openwakeword_dir / "openwakeword"
    train_py = train_dir / "train.py"
    piper_repo = config_path.parent / "piper-sample-generator"

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{piper_repo}:{openwakeword_dir}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    allowlist_logic = (
        "import torch, types; import piper_train.vits.models; "
        "torch.serialization.add_safe_globals([piper_train.vits.models.SynthesizerTrn, "
        "piper_train.vits.models.TextEncoder]); "
        "import torchaudio; import soundfile as sf; "
        "hasattr(torchaudio, 'info') or setattr(torchaudio, 'info', "
        "(lambda path, format=None, buffer_size=4096, backend=None: "
        "types.SimpleNamespace(sample_rate=sf.info(path).samplerate, "
        "num_frames=sf.info(path).frames, num_channels=sf.info(path).channels, "
        "bits_per_sample=0, encoding=getattr(sf.info(path), 'subtype', 'UNKNOWN')))); "
    )

    for flag, extra in (
        ("--generate_clips", []),
        ("--augment_clips", ["--overwrite"]),
        ("--train_model", []),
    ):
        argv = ["train.py", "--training_config", str(config_path), flag, *extra]
        cmd = [
            sys.executable,
            "-u",
            "-c",
            f"{allowlist_logic} import sys; sys.argv={argv!r}; exec(open('{train_py}').read())",
        ]
        print(f"\nRunning training stage: {flag}")
        run(cmd, cwd=train_dir, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reachy wakeword model locally.")
    parser.add_argument(
        "--work-dir",
        default="./wakeword-training",
        help="Working directory for assets/models.",
    )
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="Skip AudioSet download (uses existing wav files in audioset_16k).",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip synthetic generation/augmentation (uses existing files).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after preparing data + config.",
    )
    parser.add_argument("--audioset-clips", type=int, default=400)
    args = parser.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    openwakeword_dir = work / "openWakeWord"
    print(f"Using work dir: {work}")

    clone_clean("https://github.com/dscripka/openWakeWord.git", openwakeword_dir)
    piper_repo = prepare_piper_sample_generator(work)

    run([sys.executable, "-m", "pip", "install", "-e", str(openwakeword_dir)])
    ensure_openwakeword_resources(openwakeword_dir)
    download_feature_files(work)

    if not args.skip_background:
        collect_audioset_background(work, n_clips=args.audioset_clips)

    if not args.skip_tts:
        try:
            raw_dir = generate_raw_synthetic(work, piper_repo / "models")
        except HfHubHTTPError as exc:
            raise RuntimeError(f"Could not download Piper voices from Hugging Face: {exc}") from exc
        augment_synthetic(raw_dir, work)

    config_path = write_config(work, piper_repo)
    print(f"Training config written to: {config_path}")

    if not args.skip_train:
        run_training_stages(openwakeword_dir, config_path)

    model_path = work / MODEL_NAME / f"{MODEL_NAME}.onnx"
    if model_path.exists():
        print(f"\nDONE: {model_path}")
    else:
        print("\nTraining completed, but ONNX file was not found at:")
        print(model_path)
        sys.exit(1)


if __name__ == "__main__":
    random.seed(int(time.time()))
    main()
