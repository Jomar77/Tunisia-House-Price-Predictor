"""dataScrape.py

Backward-compatible entrypoint for the legacy scraping script.

The implementation lives in `scripts/dataScrape.py` to keep data collection
tooling separate from the runtime application.
"""


def main() -> None:
    from scripts.dataScrape import main as _main

    _main()


if __name__ == "__main__":
    main()