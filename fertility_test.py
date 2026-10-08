"""
Fertility comparison: AfriSyl vs o200k (GPT-4o's tokenizer).

"Fertility" = tokens produced / words in the input (lower is better -- it
means each token carries more linguistic content). This script also reports
tokens/byte so you can compare compression independent of what counts as a
"word".

Usage:
    pip install tiktoken
    python fertility_test.py

Note: tiktoken downloads the o200k_base merge file from OpenAI's CDN on
first use, so this needs normal internet access (it will fail in network-
restricted sandboxes that don't allowlist openaipublic.blob.core.windows.net).

Edit SAMPLE_TEXT below (or point it at a real corpus file) per language.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from afrisyl_tokenizer import AfriSylTokenizer  # noqa: E402

try:
    import tiktoken
except ImportError:
    print("pip install tiktoken first: pip install tiktoken")
    sys.exit(1)


SAMPLE_TEXT = {
    "shona": (
        "Mhoroi shamwari, ndinoda kukuudza nyaya yakatanga kare kare "
        "muZimbabwe. Vanhu vakawanda vaigara mumusha uyu, vachirima "
        "zviyo nezvimwe zvirimwa. Mwana wangu ane makore gumi nemaviri, "
        "anofunda chikoro chepamusoro. Ndakaona vanhu vaviri vachitamba "
        "nezuro manheru, vaifara zvikuru!"
    ),
    "ndebele": (
        "Sawubona mngane wami, ngifuna ukukutshela indaba eyaqala "
        "endulo eZimbabwe. Abantu abanengi babehlala kulo muzi, "
        "belima amabele lezinye izilimo. Umntanami uleminyaka "
        "elitshumi lambili, ufunda esikoleni esiphakeme."
    ),
    "tswana": (
        "Dumela tsala ya me, ke batla go go bolelela kgang e e "
        "simolotseng kgale mo Botswana. Batho ba bantsi ba ne ba "
        "dula mo motseng ono, ba lema mabele le dijalo tse dingwe. "
        "Ngwana wa me o na le dingwaga di le lesome le bobedi."
    ),
}


def fertility(text: str, tok: AfriSylTokenizer, o200k) -> None:
    words = text.split()
    n_words = len(words)
    n_bytes = len(text.encode("utf-8"))

    afrisyl_ids = tok.encode(text)
    o200k_ids = o200k.encode(text)

    print(f"  words={n_words}  bytes={n_bytes}")
    print(f"  AfriSyl : {len(afrisyl_ids):4d} tokens  "
          f"fertility={len(afrisyl_ids)/n_words:.2f} tok/word  "
          f"{len(afrisyl_ids)/n_bytes:.3f} tok/byte")
    print(f"  o200k   : {len(o200k_ids):4d} tokens  "
          f"fertility={len(o200k_ids)/n_words:.2f} tok/word  "
          f"{len(o200k_ids)/n_bytes:.3f} tok/byte")

    # sanity check: AfriSyl should round-trip exactly (o200k always does)
    roundtrip_ok = tok.decode(afrisyl_ids) == text.lower()
    print(f"  AfriSyl round-trip exact: {roundtrip_ok}")


def main():
    o200k = tiktoken.get_encoding("o200k_base")
    for lang, text in SAMPLE_TEXT.items():
        print(f"=== {lang} ===")
        tok = AfriSylTokenizer(language=lang)
        fertility(text, tok, o200k)
        print()


if __name__ == "__main__":
    main()
