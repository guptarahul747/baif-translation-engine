import argparse

from run_step1_asr import run_whisper_extraction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the production ASR pipeline")
    parser.add_argument("input_file", type=str)
    parser.add_argument(
        "--lang",
        "-l",
        choices=["auto", "mr", "hi", "en"],
        default="en",
    )
    args = parser.parse_args()

    run_whisper_extraction(args.input_file, src_lang=args.lang)
