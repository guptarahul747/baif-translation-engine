import json
import re
import sys
import time
from pathlib import Path

import ctranslate2
import sentencepiece as spm

# Resilient dual-fallback import for Domain Lexicon
try:
    from app.modules.domain_lexicon import (
        normalize_marathi_text,
        normalize_hindi_text,
        normalize_text,
    )
except ModuleNotFoundError:
    try:
        from domain_lexicon import (
            normalize_marathi_text,
            normalize_hindi_text,
            normalize_text,
        )
    except ModuleNotFoundError:
        def normalize_marathi_text(text: str) -> str:
            return text.strip() if text else ""

        def normalize_hindi_text(text: str) -> str:
            return text.strip() if text else ""

        def normalize_text(text: str, lang: str = "mr") -> str:
            return text.strip() if text else ""

BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault" / "indictrans2"
DEFAULT_INPUT = BASE_DIR / "storage_vault" / "outputs" / "step1_whisper_output.json"
OUTPUT_JSON = BASE_DIR / "storage_vault" / "outputs" / "step2_offline_translated.json"

LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
}
SUPPORTED_LANGUAGES = set(LANG_CODES)
SENTENCE_END_RE = re.compile(r"[.!?।॥][\"'”’)]*$")


def normalized_token(token: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]", "", token, flags=re.UNICODE).casefold()


def collapse_repeated_phrases(text: str, max_phrase_words: int = 8) -> str:
    """Remove consecutive repeated word groups from ASR or NMT output."""
    tokens = clean_text(text).split()
    result: list[str] = []
    index = 0

    while index < len(tokens):
        best_length = 0
        best_count = 1
        limit = min(max_phrase_words, (len(tokens) - index) // 2)
        for phrase_length in range(limit, 0, -1):
            phrase = [normalized_token(token) for token in tokens[index:index + phrase_length]]
            if not all(phrase):
                continue
            count = 1
            while index + ((count + 1) * phrase_length) <= len(tokens):
                candidate = [
                    normalized_token(token)
                    for token in tokens[
                        index + (count * phrase_length):index + ((count + 1) * phrase_length)
                    ]
                ]
                if candidate != phrase:
                    break
                count += 1
            if count > 1:
                best_length = phrase_length
                best_count = count
                break

        if best_count > 1:
            result.extend(tokens[index:index + best_length])
            index += best_length * best_count
        else:
            result.append(tokens[index])
            index += 1

    return " ".join(result)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2581", " ").replace("▁", " ")
    return re.sub(r"\s+", " ", text).strip()


def merge_segments_for_translation(
    segments: list[dict],
    min_chars_before_sentence_break: int = 90,
    max_chars: int = 280,
    max_duration_seconds: float = 18.0,
) -> list[dict]:
    """Merge adjacent Whisper fragments into sentence-like chunks before NMT."""
    merged: list[dict] = []
    buffer: list[dict] = []

    def flush():
        nonlocal buffer
        if not buffer:
            return

        source_text = clean_text(" ".join((item.get("text") or "").strip() for item in buffer))
        if source_text:
            merged.append(
                {
                    "id": len(merged) + 1,
                    "start": float(buffer[0].get("start", 0.0)),
                    "end": float(buffer[-1].get("end", buffer[0].get("start", 0.0) + 1.0)),
                    "text": source_text,
                    "source_segment_ids": [item.get("id") for item in buffer],
                }
            )
        buffer = []

    for segment in segments:
        text = clean_text(segment.get("text") or "")
        if not text:
            continue

        if not buffer:
            buffer.append(segment)
        else:
            proposed_text = clean_text(
                " ".join((item.get("text") or "").strip() for item in buffer) + " " + text
            )
            proposed_duration = float(segment.get("end", 0.0)) - float(buffer[0].get("start", 0.0))

            if len(proposed_text) > max_chars or proposed_duration > max_duration_seconds:
                flush()
            buffer.append(segment)

        current_text = clean_text(" ".join((item.get("text") or "").strip() for item in buffer))
        current_duration = float(buffer[-1].get("end", 0.0)) - float(buffer[0].get("start", 0.0))

        if (
            len(current_text) >= min_chars_before_sentence_break
            and SENTENCE_END_RE.search(current_text)
        ) or len(current_text) >= max_chars or current_duration >= max_duration_seconds:
            flush()

    flush()
    return merged


class NativeIndicTranslator:
    """CTranslate2 + direction-specific SentencePiece tokenizer."""

    def __init__(self, model_dir: Path, name: str):
        self.model_dir = model_dir
        self.name = name

        model_bin = model_dir / "model.bin"
        src_spm = model_dir / "vocab" / "model.SRC"
        tgt_spm = model_dir / "vocab" / "model.TGT"
        src_vocab = model_dir / "source_vocabulary.json"

        for path in (model_bin, src_spm, tgt_spm, src_vocab):
            if not path.exists():
                raise FileNotFoundError(f"Required translation asset missing: {path}")

        print(f"⏳ Loading {name} CTranslate2 model...", flush=True)
        t0 = time.perf_counter()
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            inter_threads=1,
            intra_threads=0,
        )
        print(f"✅ {name} model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

        self.sp_src = spm.SentencePieceProcessor()
        self.sp_src.load(str(src_spm))
        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_tgt.load(str(tgt_spm))

        vocab_data = json.loads(src_vocab.read_text(encoding="utf-8"))
        self.src_vocab = set(vocab_data if isinstance(vocab_data, list) else vocab_data.keys())

    def get_lang_tag(self, language: str) -> str:
        code = LANG_CODES[language]
        candidates = [code, f"_{code}_", f"__{code}__", f"__{code}"]
        return next((candidate for candidate in candidates if candidate in self.src_vocab), code)

    def translate_texts(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if not texts:
            return []

        src_tag = self.get_lang_tag(src_lang)
        tgt_tag = self.get_lang_tag(tgt_lang)
        tokenized_inputs = []

        for text in texts:
            pieces = self.sp_src.encode((text or "").strip(), out_type=str)
            tokenized_inputs.append([src_tag, tgt_tag] + pieces)

        print(
            f"🌐 {src_lang.upper()} → {tgt_lang.upper()} | "
            f"{len(tokenized_inputs)} chunk(s) using {self.name} | beam=4",
            flush=True,
        )
        t0 = time.perf_counter()
        results = self.translator.translate_batch(
            tokenized_inputs,
            beam_size=4,
            max_decoding_length=256,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
        )
        print(f"✅ Translation pass finished in {time.perf_counter() - t0:.2f}s", flush=True)

        translations = []
        known_tags = set(LANG_CODES.values())
        for result in results:
            out_tokens = result.hypotheses[0]
            clean_tokens = []
            for token in out_tokens:
                if token in {src_tag, tgt_tag, "<s>", "</s>", "<unk>", "<pad>"}:
                    continue
                if token in known_tags:
                    continue
                if token.startswith("__") and token.endswith("__"):
                    continue
                clean_tokens.append(token)

            translated = self.sp_tgt.decode_pieces(clean_tokens)
            post_processed = collapse_repeated_phrases(translated)

            # Target-side normalization to eliminate NMT hallucinations
            if tgt_lang == "hi":
                post_processed = normalize_hindi_text(post_processed)
            elif tgt_lang == "mr":
                post_processed = normalize_marathi_text(post_processed)

            translations.append(clean_text(post_processed))

        return translations


def build_engines(src_lang: str, tgt_lang: str):
    en_indic = None
    indic_en = None

    needs_en_indic = src_lang == "en" or (src_lang in {"hi", "mr"} and tgt_lang in {"hi", "mr"})
    needs_indic_en = tgt_lang == "en" or (src_lang in {"hi", "mr"} and tgt_lang in {"hi", "mr"})

    if needs_en_indic:
        en_indic = NativeIndicTranslator(VAULT_DIR / "en-indic", "en-indic")
    if needs_indic_en:
        indic_en = NativeIndicTranslator(VAULT_DIR / "indic-en", "indic-en")

    return en_indic, indic_en


def translate_texts_all_routes(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    en_indic: NativeIndicTranslator | None,
    indic_en: NativeIndicTranslator | None,
) -> list[str]:
    if src_lang == tgt_lang:
        return [clean_text(text) for text in texts]

    if src_lang == "en" and tgt_lang in {"hi", "mr"}:
        return en_indic.translate_texts(texts, "en", tgt_lang)

    if src_lang in {"hi", "mr"} and tgt_lang == "en":
        return indic_en.translate_texts(texts, src_lang, "en")

    if src_lang in {"hi", "mr"} and tgt_lang in {"hi", "mr"}:
        print(f"🔁 Pivot route: {src_lang.upper()} → EN → {tgt_lang.upper()}", flush=True)
        english = indic_en.translate_texts(texts, src_lang, "en")
        return en_indic.translate_texts(english, "en", tgt_lang)

    raise ValueError(f"Unsupported translation pair: {src_lang} -> {tgt_lang}")


def translate_whisper_json(
    input_json_path: str,
    src_lang: str = "en",
    target_lang: str = "hi",
):
    src_lang = src_lang.lower()
    target_lang = target_lang.lower()

    if src_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError("Supported language codes are: en, hi, mr")
    if src_lang == target_lang:
        raise ValueError("Source and target languages must be different")

    input_path = Path(input_json_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    raw_segments = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_segments, list):
        raise ValueError("Input JSON must contain a list of transcript segments")

    print("=" * 64, flush=True)
    print("🌐 STAGE 2: OFFLINE MACHINE TRANSLATION — DOMAIN-AWARE MODE", flush=True)
    print("=" * 64, flush=True)
    print(f"Route requested: {src_lang.upper()} → {target_lang.upper()}", flush=True)
    print(f"Raw ASR segments: {len(raw_segments)}", flush=True)

    # Pre-translation normalization: clean phonetic ASR errors & remove corrupted scripts
    for seg in raw_segments:
        raw_text = seg.get("text", "")
        if src_lang == "mr":
            seg["text"] = normalize_marathi_text(raw_text)
        elif src_lang == "hi":
            seg["text"] = normalize_hindi_text(raw_text)

    # Merge adjacent fragments into contextual sentences
    if len(raw_segments) > 1:
        segments = merge_segments_for_translation(raw_segments)
    else:
        segments = raw_segments

    print(f"Translation chunks after context merge: {len(segments)}", flush=True)
    for idx, seg in enumerate(segments[:5], start=1):
        print(f"  SRC[{idx}]: {seg.get('text', '')}", flush=True)
    if len(segments) > 5:
        print(f"  ... {len(segments) - 5} more chunk(s)", flush=True)

    texts = [collapse_repeated_phrases(segment.get("text") or "") for segment in segments]
    en_indic, indic_en = build_engines(src_lang, target_lang)

    t0 = time.perf_counter()
    translated_texts = translate_texts_all_routes(
        texts,
        src_lang,
        target_lang,
        en_indic,
        indic_en,
    )

    translated_segments = []
    for segment, translated in zip(segments, translated_texts):
        entry = dict(segment)
        entry[f"translation_{target_lang}"] = translated
        entry["translated_text"] = translated
        translated_segments.append(entry)
        print(
            f"  {src_lang.upper()}: {entry.get('text', '')}\n"
            f"  {target_lang.upper()}: {translated}\n",
            flush=True,
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(translated_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Complete route finished in {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"📄 Output: {OUTPUT_JSON}", flush=True)
    print("=" * 64, flush=True)
    return OUTPUT_JSON


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    target_language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    source_language = sys.argv[3] if len(sys.argv) > 3 else "en"

    translate_whisper_json(
        input_file,
        src_lang=source_language,
        target_lang=target_language,
    )