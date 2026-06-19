from __future__ import annotations

import argparse
import os
import pandas as pd
import re
import unicodedata
from pathlib import Path


SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _normalize_determinative(match):
    content = match.group(1).strip().lower()
    content = re.sub(r"[@~]v", "", content)
    if content in ("1", "m", "disz"):
        content = "m"
    elif content in ("mi2", "f", "munus"):
        content = "f"
    elif content in ("iri", "uru"):
        content = "uru"
    return "{" + content + "}"


def _normalize_subscripts(text: str) -> str:
    def replace(match):
        base = match.group(1)
        index = match.group(2).replace("_", "").translate(SUBSCRIPT_DIGITS)
        return base + index

    return re.sub(r"(?<!\\w)([A-Za-zšṣṭḫĝŋŠṢṬḪ]+)_?([0-9]+)", replace, text)


def _normalize_akkadian_gaps(text: str) -> str:
    text = re.sub(r"\.{4,}", " <big_gap> ", text)
    text = re.sub(r"\.{3}", " <gap> ", text)
    text = re.sub(r"\[(?:\s*\.){0,3}\s*\]", " <gap> ", text)
    text = re.sub(r"\b(?:x|X)\b", " <gap> ", text)
    text = re.sub(r"(?<!\\w)\[\s*\]\b", " <gap> ", text)
    text = re.sub(r"(?<!\\w)\[\.{1,3}\](?!\\w)", " <gap> ", text)
    text = re.sub(r"(?<!\\w)\[\.\.\](?!\\w)", " <gap> ", text)
    return text


def _strip_scholarly_noise(text: str) -> str:
    text = re.sub(r"(?m)^\s*\d+\.\s*", "", text)
    text = re.sub(r"(?m)^\s*[ivxlcdm]+\s+\d+\'?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d+\s+\d+\'?\s*", "", text)
    text = re.sub(r"\{([^{}]+?)(?:[@~](?:v|obverse|reverse|obv|rev))\}", lambda m: "{" + m.group(1) + "}", text, flags=re.IGNORECASE)
    text = re.sub(r"([^\s{}]+?)(?:[@~](?:v|obverse|reverse|obv|rev))\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\s*@(?:obverse|reverse|obv|rev)\b", " ", text)
    text = re.sub(r"(?<!\\w)[#!?]+(?!\\w)", "", text)
    text = re.sub(r"(?i)\bo\s*$", "", text)
    return text


def normalize_akkadian(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _normalize_subscripts(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\{([^}]+)\}", _normalize_determinative, text)
    text = _strip_scholarly_noise(text)
    text = _normalize_akkadian_gaps(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_english(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\.{4,}", " <big_gap> ", text)
    text = re.sub(r"\.{3}", " <gap> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_identifiable_signs(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"<big_gap>|<gap>|\{[^}]+\}|[^\s]+", text)
    sign_count = 0
    for token in tokens:
        if token in {"<gap>", "<big_gap>"}:
            continue
        if token.startswith("{") and token.endswith("}"):
            sign_count += 1
            continue
        parts = [part for part in re.split(r"[-\s]+", token) if part and part not in {"#", "!", "?", "o"}]
        sign_count += len(parts)
    return sign_count


def is_meaningful_pair(source_text: str, target_text: str, min_id_sign_count: int = 3, max_length_ratio: float = 3.0) -> bool:
    if not source_text or not target_text:
        return False
    source_core = source_text.replace("<gap>", "").replace("<big_gap>", "").strip()
    if not source_core:
        return False
    if count_identifiable_signs(source_text) < min_id_sign_count:
        return False
    source_length = len(source_text.split())
    target_length = len(target_text.split())
    if source_length == 0 or target_length == 0:
        return False
    length_ratio = max(source_length / target_length, target_length / source_length)
    return length_ratio <= max_length_ratio


def build_block_sizes(total_rows: int, min_lines: int = 2, max_lines: int = 3) -> list[int]:
    if total_rows < min_lines:
        return []
    if total_rows == 2:
        return [2]
    remainder = total_rows % max_lines
    if remainder == 0:
        return [3] * (total_rows // 3)
    if remainder == 1:
        if total_rows < 4:
            return []
        return [3] * (total_rows // 3 - 1) + [2, 2]
    return [3] * (total_rows // 3) + [2]


def group_into_blocks(frame: pd.DataFrame, min_lines: int = 2, max_lines: int = 3) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["akkadian", "english"])
    block_sizes = build_block_sizes(len(frame), min_lines, max_lines)
    grouped_rows = []
    start_index = 0
    for block_size in block_sizes:
        block = frame.iloc[start_index:start_index + block_size]
        source_text = " ".join(block["source_text"].tolist()).strip()
        target_text = " ".join(block["target_text"].tolist()).strip()
        if is_meaningful_pair(source_text, target_text):
            grouped_rows.append({"akkadian": source_text, "english": target_text})
        start_index += block_size
    return pd.DataFrame(grouped_rows, columns=["akkadian", "english"])


def load_clean_parallel_dataset(source_path: Path, target_path: Path, save_path: Path | None = None, use_blocks: bool = True) -> pd.DataFrame:
    cleaned_rows = []
    with open(source_path, encoding="utf-8") as source_file, open(target_path, encoding="utf-8") as target_file:
        for source_line, target_line in zip(source_file, target_file):
            source_text = normalize_akkadian(source_line.rstrip("\n"))
            target_text = normalize_english(target_line.rstrip("\n"))
            if is_meaningful_pair(source_text, target_text):
                cleaned_rows.append({"source_text": source_text, "target_text": target_text})
    cleaned_frame = pd.DataFrame(cleaned_rows, columns=["source_text", "target_text"])
    if use_blocks:
        final_frame = group_into_blocks(cleaned_frame)
    else:
        final_frame = cleaned_frame.rename(columns={"source_text": "akkadian", "target_text": "english"})
    if save_path is not None:
        final_frame.to_csv(save_path, index=False)
    return final_frame


def find_input_files(root: Path) -> dict[str, Path]:
    candidates = {
        "train_src": [root / "transcription_train.txt", root / "akkadian_train.txt", root / "train.txt"],
        "train_tgt": [root / "english_train.txt", root / "train_en.txt"],
        "val_src": [root / "transcription_validation.txt", root / "akkadian_validation.txt", root / "validation.txt"],
        "val_tgt": [root / "english_validation.txt", root / "validation_en.txt"],
    }
    found = {}
    for key, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[key] = p
                break
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / ".." / "orignal" / "data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / ".." / "data")
    parser.add_argument("--no-blocks", action="store_true")
    args = parser.parse_args()

    src_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = find_input_files(src_dir)
    if not inputs.get("train_src") or not inputs.get("train_tgt"):
        raise FileNotFoundError(f"Could not find train source/target files in {src_dir}")

    use_blocks = not args.no_blocks

    train_out = out_dir / "train_cleaned.csv"
    val_out = out_dir / "validation_cleaned.csv"
    test_out = out_dir / "test_cleaned.csv"

    print(f"Reading from {src_dir}")
    df_train = load_clean_parallel_dataset(inputs["train_src"], inputs["train_tgt"], save_path=train_out, use_blocks=use_blocks)
    print(f"Saved cleaned train to {train_out} ({len(df_train)} rows)")

    if inputs.get("val_src") and inputs.get("val_tgt"):
        df_val = load_clean_parallel_dataset(inputs["val_src"], inputs["val_tgt"], save_path=val_out, use_blocks=use_blocks)
        print(f"Saved cleaned validation to {val_out} ({len(df_val)} rows)")
    else:
        print("Validation files not found; skipping validation preprocessing.")

    # copy test to test_cleaned if present (some datasets have separate test files)
    if (src_dir / "test.txt").exists():
        df_test = load_clean_parallel_dataset(src_dir / "test.txt", src_dir / "test_en.txt", save_path=test_out, use_blocks=use_blocks)
        print(f"Saved cleaned test to {test_out} ({len(df_test)} rows)")


if __name__ == "__main__":
    main()
