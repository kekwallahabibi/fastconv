---
name: prove-it-works
description: Verwenden vor jedem Abschluss eines Features, Bugfixes oder Performance-Tunings in fastconv, um Korrektheit, Durchsatz und Hardware-Ausführung objektiv zu belegen.
---

# Prove It Works (Verifikations- & Hillclimbing-Protokoll)

Kein Task gilt als abgeschlossen, solange nicht jeder relevante Nachweis durch echte Terminal-Ausgaben belegt ist.

## 1. Statische & Test-Verifikation (Pflicht)

Führe im Repository-Root aus:

```bash
cargo xtask verify
```

Alle drei Stufen müssen fehlerfrei grün sein:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`

## 2. Echte Konvertierung gegen `fixtures/` prüfen

Wenn Konvertierungs-, Probe- oder Pipeline-Logik geändert wurde:

1. Führe die echte `fastconv`-Binary gegen mindestens eine reale Eingabedatei aus `fixtures/` aus (Ausgabe in einen temporären, ignorierten Pfad schreiben).
2. Verifiziere die erzeugte Ausgabedatei objektiv (Existenz, nicht leer, Container, Codec, Auflösung/Sample-Rate, Dauer, Streams — z. B. via `ffprobe` oder Parser-Output).
3. Prüfe Fehlerpfade explizit (z. B. fehlende Quelldatei, identischer Quell-/Zielpfad, inkompatibles Backend).

## 3. Durchsatz-Hillclimbing gegen Baseline

Bei Performance-Tuning oder neuen Backends:

1. Miss vor der Änderung eine reproduzierbare Baseline auf denselben `fixtures/` (z. B. via `cargo xtask bench` oder definiertem Zeit-/Durchsatz-Kommando).
2. Miss nach der Änderung exakt dasselbe Szenario und vergleiche:
   - Laufzeit (ms / s)
   - Durchsatz (MB/s)
   - Verarbeitungsrate (FPS)
3. Zeigt eine Optimierung keinen messbaren Gewinn oder verschlechtert sie Klarheit ohne spürbaren Speedup, verwirf sie.

## 4. Beweis gegen stillen CPU-Fallback

Bei jeder GPU-Konvertierung ist explizit nachzuweisen, dass kein stiller CPU-Fallback stattfand:

1. Belege den gewählten `ExecutionPlan` (`GpuZeroCopy` mit konkretem `GpuBackend`).
2. Prüfe Laufzeit-Nachweise (Treiber-/Encoder-Telemetrie, `nvidia-smi`/VAAPI-/VideoToolbox-Metriken oder explizite Encoder-Logs).
3. Ist die angeforderte GPU lokal nicht verfügbar, muss der Lauf sauber mit Fehler abbrechen — niemals still auf CPU wechseln.

## 5. Abschluss-Nachweis

Dokumentiere in der Antwort (und später im PR):
- exakte ausgeführte Kommandos,
- Ergebnis von `cargo xtask verify`,
- gemessene Baseline vs. Nachher-Werte (FPS, MB/s, Laufzeit) und Ziel-Format-Check,
- explizite Angabe, welche Hardware-Checks mangels lokaler GPU ggf. nicht direkt ausführbar waren.
