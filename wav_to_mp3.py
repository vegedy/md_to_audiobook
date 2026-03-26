from pathlib import Path
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3

AUDIO_DIR = Path("audio_kapitel")
ALBUM_NAME = "Deep Learning Vortrag"
ARTIST_NAME = "Benito"
YEAR = "2026"

def convert_wav_to_mp3_with_tags(audio_dir: Path):
    wav_files = sorted(audio_dir.glob("*.wav"))
    for track_num, wav_path in enumerate(wav_files, start=1):
        mp3_path = wav_path.with_suffix(".mp3")

        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")

        tags = EasyID3()
        tags["title"] = [wav_path.stem]
        tags["album"] = [ALBUM_NAME]
        tags["artist"] = [ARTIST_NAME]
        tags["tracknumber"] = [str(track_num)]
        tags["date"] = [YEAR]
        tags.save(mp3_path)

        print(f"MP3 erzeugt: {mp3_path}")

if __name__ == "__main__":
    convert_wav_to_mp3_with_tags(AUDIO_DIR)
    print("Alle MP3s mit Album/Track-Tags fertig.")
