from pathlib import Path
import torch

_original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

from TTS.api import TTS

INPUT_MD = Path("data/Deep Learning – Vollständiger Vortrag auf M.Sc.-Niveau.md")
OUT_DIR = Path("audio_kapitel")
OUT_DIR.mkdir(exist_ok=True)

# MODEL_NAME = "tts_models/de/thorsten/tacotron2-DCA"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# MODEL_NAME = "tts_models/de/thorsten/vits"
tts = TTS(MODEL_NAME)
tts.to("cuda")

# ... Rest (Markdown split + tts.tts_to_file) ...

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
        tts.tts_to_file(
            text=content,
            file_path=str(out_path),
            speaker_wav=None,   # oder Pfad zu einer Referenz-WAV
            speaker="Daisy Studious",  # oder ein anderer vordefinierter Sprecher
            language="de",
        )
        wav_paths.append(out_path)
        # exit(0)  # Debug: Nur ein Kapitel testen

    return wav_paths

if __name__ == "__main__":
    wav_files = synthesize_chapters(INPUT_MD, OUT_DIR)
    print("Fertig, Kapitel-WAVs:", wav_files)
