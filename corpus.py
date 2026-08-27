import gzip
import requests
import re  # back to normal re
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CORPUS_FILE = DATA_DIR / "corpus.txt"

SOURCES = {
    "jw300": "https://object.pouta.csc.fi/OPUS-JW300/v1.0/moses/sh/xx.txt.gz",
    "opus_sha": "https://object.pouta.csc.fi/OPUS-OSCAR/202309/sha.txt.gz",
    "opus_sha_dedup": "https://object.pouta.csc.fi/OPUS-OSCAR/202309/sha-dedup.txt.gz",
}

def download_file(name, url):
    out_path = DATA_DIR / f"{name}.txt.gz"
    if out_path.exists():
        return out_path
    print(f"[DL] {name}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192), desc=name):
            f.write(chunk)
    return out_path

def read_file(path):
    sentences = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                sentences.append(line.strip())
    return sentences

def clean_text(text: str) -> str:
    """Shona specific cleaning"""
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text) # urls
    text = re.sub(r'\d+', '', text) # numbers
    # Keep only a-z + shona digraphs + basic punctuation + space
    # We keep the letters and let the tokenizer handle syllables later
    text = re.sub(r'[^a-z\s,.!?;:\'\"-]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text.split()) > 2 else ""

def build_corpus():
    all_sentences = []
    
    for name, url in SOURCES.items():
        path = download_file(name, url)
        all_sentences.extend(read_file(path))
    
    # Local files
    local_dir = DATA_DIR / "sources"
    if local_dir.exists():
        for txt_file in local_dir.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                all_sentences.extend([l.strip() for l in f if l.strip()])
    
    print("[CLEAN] Cleaning and deduping...")
    cleaned = [clean_text(s) for s in all_sentences]
    cleaned = [s for s in cleaned if s]
    unique = list(dict.fromkeys(cleaned))
    
    print(f"[SAVE] {len(unique)} lines -> {CORPUS_FILE}")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(unique))
    
    print("✅ Done")

if __name__ == "__main__":
    build_corpus()
