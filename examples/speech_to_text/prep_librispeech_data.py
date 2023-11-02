#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import COMMONVOICE
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = [
        "train",
        "dev",
        "test",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    args.output_root = args.output_root.split(",")
    final_root = args.output_root[-1]
    final_root = Path(final_root).absolute()
    final_root.mkdir(exist_ok=True)
    if len(args.output_root) > 1:
        args.output_root = args.output_root[:-1]
    for out_root in args.output_root:
        out_root = Path(out_root).absolute()
        out_root.mkdir(exist_ok=True)
        # Extract features
        feature_root = final_root / "fbank80"
        feature_root.mkdir(exist_ok=True)
        for split in SPLITS:
            lang = str(out_root).split("/")[-1]
            if split != "train" and lang != "gn":
                continue
            print(f"Fetching split {split} for data @ {out_root}...")
            dataset = COMMONVOICE(out_root.as_posix(), tsv=split + ".tsv")
            print("Extracting log mel filter bank features...")
            # for wav, sample_rate, _, spk_id, chapter_no, utt_no in tqdm(dataset):
            data_items = []
            for i in range(len(dataset)):
                try:
                    item = dataset[i]
                    data_items.append(item)
                except Exception as e:
                    print(f"[{i}] {e}")
            for wav, sample_rate, dic in tqdm(dataset):
                # sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                sample_id = dic["path"]
                extract_fbank_features(
                    wav, sample_rate, feature_root / f"{sample_id}.npy"
                )
        # Pack features into ZIP
        zip_path = out_root / "fbank80.zip"
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
        print("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(zip_path)
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        for split in SPLITS:
            lang = str(out_root).split("/")[-1]
            if split != "train" and lang != "gn":
                continue
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = COMMONVOICE(out_root.as_posix(), tsv=split + ".tsv")
            # for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            data_items = []
            for i in range(len(dataset)):
                try:
                    item = dataset[i]
                    data_items.append(item)
                except Exception as e:
                    print(f"[{i}] {e}")
            for wav, sample_rate, dic in tqdm(data_items):
                # sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                sample_id = dic["path"]
                manifest["id"].append(sample_id)
                manifest["audio"].append(audio_paths[sample_id])
                manifest["n_frames"].append(audio_lengths[sample_id])
                # manifest["tgt_text"].append(utt.lower())
                manifest["tgt_text"].append(dic["sentence"])
                # manifest["speaker"].append(spk_id)
                manifest["speaker"].append(dic["client_id"])
            save_df_to_tsv(
                pd.DataFrame.from_dict(manifest), final_root / f"{split}_manifest_{lang}.tsv"
            )
            if split.startswith("train"):
            # if split.startswith("validated"):
                train_text.extend(manifest["tgt_text"])
    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    # bytes_written = 0
    with NamedTemporaryFile(mode="w") as f:
        for train_idx, t in enumerate(train_text):
            # if train_idx == 1177:
            #     continue
            f.write(t + "\n")
        f.flush()
        # import pdb; pdb.set_trace()
        gen_vocab(
            Path(f.name),
            final_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        final_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
