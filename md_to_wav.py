from pathlib import Path
import torch

_original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

from TTS.api import TTS
import re
from pydub import AudioSegment
from pydub import effects
from pysbd import Segmenter
import tempfile


INPUT_MD = Path("data/Shortcut Learning in Deep Neural Networks.md")
OUT_DIR = Path("audio_kapitel")
OUT_DIR.mkdir(exist_ok=True)

# MODEL_NAME = "tts_models/de/thorsten/tacotron2-DCA"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# MODEL_NAME = "tts_models/de/thorsten/vits"
tts = TTS(MODEL_NAME)
tts.to("cuda")

MAX_CHARS = 250

SILENCE_BETWEEN_CHUNKS = 200  # ms pause between chunks
SPEAKER_WAV = Path("data/christian-brueckner.wav") # Optional: Path to a speaker reference audio file
AUTHOR = "Perplexity AI"

_segmenter = Segmenter(language="de", clean=True)

def split_into_sentences(text: str):
    return _segmenter.segment(text)

def split_long_sentence(sent: str, max_chars: int = MAX_CHARS):
    if len(sent) <= max_chars:
        return [sent]
    chunks = []
    current = []
    current_len = 0
    for token in sent.split():
        if current_len + len(token) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len += len(token) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

def text_to_chunks(text: str, max_chars: int = MAX_CHARS):
    chunks = []
    for sent in split_into_sentences(text):
        chunks.extend(split_long_sentence(sent, max_chars))
    return chunks

MARKDOWN_PATTERNS = [
    (re.compile(r'^\s*[-*_]{3,}\s*$', re.MULTILINE), ''),
    (re.compile(r'!\[.*?\]\(.*?\)'), ''),
    (re.compile(r'\[\^.*?\]'), ''),
    (re.compile(r'\[([^\]]*)\]\(.*?\)'), r'\1'),
    (re.compile(r'```.+?```', re.DOTALL), ''),
    (re.compile(r'`([^`]+)`'), r'\1'),
    (re.compile(r'\*\*(.+?)\*\*'), r'\1'),
    (re.compile(r'__(.+?)__'), r'\1'),
    (re.compile(r'\*(.+?)\*'), r'\1'),
    (re.compile(r'_(.+?)_'), r'\1'),
    (re.compile(r'^>\s?', re.MULTILINE), ''),
    (re.compile(r'^#{1,6}\s+', re.MULTILINE), ''),
    (re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE), ''),
    (re.compile(r'^[\s]*\d+\.\s+', re.MULTILINE), ''),
    (re.compile(r'<[^>]+>'), ''),
]

def strip_markdown(text: str) -> str:
    for pattern, replacement in MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

ABBREVIATIONS = [
    (re.compile(r'\bz\.\s*B\.'), 'zum Beispiel'),
    (re.compile(r'\bbzw\.'), 'beziehungsweise'),
    (re.compile(r'\bDr\.'), 'Doktor'),
    (re.compile(r'\bProf\.'), 'Professor'),
    (re.compile(r'\bca\.'), 'circa'),
    (re.compile(r'\betc\.'), 'et cetera'),
    (re.compile(r'\bd\.\s*h\.'), 'das heißt'),
    (re.compile(r'\bu\.\s*a\.'), 'unter anderem'),
    (re.compile(r'\bvgl\.'), 'vergleiche'),
    (re.compile(r'\bggf\.'), 'gegebenenfalls'),
    (re.compile(r'\bz\.\s*Z\.'), 'zur Zeit'),
    (re.compile(r'\bs\.'), 'siehe'),
    (re.compile(r'\bNr\.'), 'Nummer'),
    (re.compile(r'\binkl\.'), 'inklusive'),
    (re.compile(r'\bexkl\.'), 'exklusive'),
    (re.compile(r'\bMio\.'), 'Millionen'),
    (re.compile(r'\bMrd\.'), 'Milliarden'),
    (re.compile(r'\bod\.'), 'oder'),
    (re.compile(r'\bsog\.'), 'sogenannt'),
    (re.compile(r'\busw\.'), 'und so weiter'),
    (re.compile(r'\bz\.\s*T\.'), 'zum Teil'),
]

def normalize_text(text: str) -> str:
    text = text.replace('\u2013', ',')
    text = text.replace('\u2014', ',')
    text = text.replace('\u2026', '...')
    for pattern, replacement in ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s+', ', ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def synthesize_text(text: str) -> AudioSegment:
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    tts.tts_to_file(
        text=text,
        file_path=tmp_wav.name,
        speaker_wav=str(SPEAKER_WAV) if SPEAKER_WAV else None,
        language="de",
    )
    return AudioSegment.from_wav(tmp_wav.name)

def synthesize_chapter(content: str, out_path, prepend=None, append=None):
    content = strip_markdown(content)
    content = normalize_text(content)
    chunks = text_to_chunks(content, MAX_CHARS)
    combined = AudioSegment.silent(duration=0)

    for i, chunk in enumerate(chunks, start=1):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()

        tts.tts_to_file(
            text=chunk,
            file_path=tmp_wav.name,
            speaker_wav=str(SPEAKER_WAV) if SPEAKER_WAV else None,
            # speaker="Daisy Studious",
            language="de",
        )

        audio = AudioSegment.from_wav(tmp_wav.name)
        if i > 1:
            combined += AudioSegment.silent(duration=SILENCE_BETWEEN_CHUNKS)
        combined += audio

    if prepend is not None:
        combined = prepend + AudioSegment.silent(duration=SILENCE_BETWEEN_CHUNKS) + combined
    if append is not None:
        combined = combined + AudioSegment.silent(duration=SILENCE_BETWEEN_CHUNKS) + append

    combined = effects.normalize(combined, headroom=3.0)

    combined.export(out_path, format="wav")

def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def split_by_h2_sections(md_text: str):
    """
    Nimmt ein Markdown mit:
    # Haupttitel
    ## Kapitel 1
    ### Unterkapitel
    ...
    und gibt eine Liste (kapitel_titel, kapitel_text) zurueck.
    """
    lines = md_text.splitlines()

    main_title = None
    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        if line.startswith("# ") and not line.startswith("##"):
            if main_title is None:
                main_title = line.lstrip("#").strip()
            continue

        if line.startswith("## "):
            if current_title is not None and current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = line.lstrip("#").strip()

        else:
            if current_title is not None:
                current_lines.append(line)

    if current_title is not None and current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return main_title, sections

def slugify(title: str) -> str:
    import re
    t = title.lower()
    t = re.sub(r"[^a-z0-9äöüß]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "kapitel"

def synthesize_chapters(md_path: Path, out_dir: Path):
    text = load_markdown(md_path)
    main_title, sections = split_by_h2_sections(text)

    print(f"Haupttitel: {main_title}")

    intro = None
    outro = None
    if main_title:
        print("[*] Synthese Intro")
        title_audio = synthesize_text(main_title)
        intro = title_audio + AudioSegment.silent(duration=2000)
        outro_text = f"Sie hörten: {main_title}, geschrieben von {AUTHOR}"
        outro = AudioSegment.silent(duration=2000) + synthesize_text(outro_text)

    wav_paths = []
    total = len(sections)

    for idx, (title, content) in enumerate(sections, start=1):
        if not content.strip():
            continue

        filename = f"{idx:02d}_{slugify(title)}.wav"
        out_path = out_dir / filename

        print(f"[*] Synthese Kapitel {idx}: {title}")
        prepend = intro if idx == 1 else None
        append = outro if idx == total else None
        synthesize_chapter(content, out_path, prepend=prepend, append=append)
        wav_paths.append(out_path)
        # exit(0)  # Debug: Nur ein Kapitel testen

    return wav_paths

if __name__ == "__main__":
    wav_files = synthesize_chapters(INPUT_MD, OUT_DIR)
    print("Fertig, Kapitel-WAVs:", wav_files)
