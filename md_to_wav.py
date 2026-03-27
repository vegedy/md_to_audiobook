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
import tempfile


INPUT_MD = Path("data/Deep Learning – Vollständiger Vortrag auf M.Sc.-Niveau.md")
OUT_DIR = Path("audio_kapitel")
OUT_DIR.mkdir(exist_ok=True)

# MODEL_NAME = "tts_models/de/thorsten/tacotron2-DCA"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# MODEL_NAME = "tts_models/de/thorsten/vits"
tts = TTS(MODEL_NAME)
tts.to("cuda")

MAX_CHARS = 230  # etwas unter 253 bleiben

sentence_end_re = re.compile(r'([.!?]+)["“”\']?\s+')

def split_into_sentences(text: str):
    parts = []
    start = 0
    for match in sentence_end_re.finditer(text):
        end = match.end()
        sent = text[start:end].strip()
        if sent:
            parts.append(sent)
        start = end
    last = text[start:].strip()
    if last:
        parts.append(last)
    return parts

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

def synthesize_chapter(content: str, out_path):
    chunks = text_to_chunks(content, MAX_CHARS)
    combined = AudioSegment.silent(duration=0)

    for i, chunk in enumerate(chunks, start=1):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()

        tts.tts_to_file(
            text=chunk,
            file_path=tmp_wav.name,
            speaker_wav=None,
            speaker="Daisy Studious",
            language="de",
        )

        audio = AudioSegment.from_wav(tmp_wav.name)
        combined += audio
        combined += AudioSegment.silent(duration=200)  # kleine Pause zwischen Chunks

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
    und gibt eine Liste (kapitel_titel, kapitel_text) zurück.
    """
    lines = md_text.splitlines()

    main_title = None
    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        # Haupttitel (# ...) nur einmal ganz oben mitnehmen, aber kein eigenes Kapitel
        if line.startswith("# ") and not line.startswith("##"):
            if main_title is None:
                main_title = line.lstrip("#").strip()
            continue

        # Kapitel: genau "## " am Zeilenanfang
        if line.startswith("## "):
            # altes Kapitel abschließen
            if current_title is not None and current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = line.lstrip("#").strip()  # "## " entfernen

        else:
            # alles (inkl. ### usw.) gehört zum aktuellen Kapitel
            if current_title is not None:
                current_lines.append(line)

    # letztes Kapitel
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
    wav_paths = []

    for idx, (title, content) in enumerate(sections, start=1):
        if not content.strip():
            continue

        filename = f"{idx:02d}_{slugify(title)}.wav"
        out_path = out_dir / filename

        print(f"[*] Synthese Kapitel {idx}: {title}")
        synthesize_chapter(content, out_path)
        wav_paths.append(out_path)
        # exit(0)  # Debug: Nur ein Kapitel testen

    return wav_paths

if __name__ == "__main__":
    wav_files = synthesize_chapters(INPUT_MD, OUT_DIR)
    print("Fertig, Kapitel-WAVs:", wav_files)
