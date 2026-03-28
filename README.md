
# Automatische Generierung eines Hörbuchs auf Basis einer markdown Datei

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Kleines Projekt welches ich nutze, um Forschungsberichte aus Perplexity
in Hörbücher zu konvertieren.

## Features

- .md Dateien in Kapitel trennen
- Lange Sätze splitten
- Je Kapitel eine wav Datei erzeugen
- wav Dateien in mp3 konvertieren und Metadaten hinzufügen

## Installation

```bash
git clone git@github.com:vegedy/md_to_audiobook.git
cd md_to_audiobook
python -m venv .venv
source .venv/bin/activate  # unter Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dieses Projekt lädt das Modell
[coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) von Hugging Face
(tts_models/multilingual/multi-dataset/xtts_v2), lizenziert unter der
[coqui-public-model-license](https://coqui.ai/cpml) – Contents: © Original
Authors.


