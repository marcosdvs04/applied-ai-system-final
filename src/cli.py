"""
Command line interface.

  python -m src.cli "music for studying late at night"
  python -m src.cli --llm "something sad and rainy"     (needs GEMINI_API_KEY, optional)
  python -m src.cli                                      (interactive; type 'quit' to exit)
"""

import argparse

from src.pipeline import VibeMatchAssistant, format_response


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeMatch: describe a vibe, get songs.")
    parser.add_argument("query", nargs="?", help="what you want to listen to")
    parser.add_argument("-k", type=int, default=5, help="number of songs (default 5)")
    parser.add_argument("--llm", action="store_true", help="polish the explanation with Gemini (optional)")
    args = parser.parse_args()

    assistant = VibeMatchAssistant(use_llm=args.llm)
    if args.query is not None:
        print(format_response(assistant.ask(args.query, k=args.k)))
        return
    print("VibeMatch. Describe what you want to hear ('quit' to exit).")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        print(format_response(assistant.ask(text, k=args.k)))
        print()


if __name__ == "__main__":
    main()
