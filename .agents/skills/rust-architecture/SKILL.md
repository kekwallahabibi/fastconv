---
name: rust-architecture
description: Verwenden beim Entwerfen neuer Crates, Domain-Typen, CPU/GPU-Pipeline-Backends oder größerer Refactorings in fastconv.
---

# Rust-Architektur (Domain-First & Laziness Protocol)

Führe diese Checkliste strikt in dieser Reihenfolge aus, bevor Implementierungscode entsteht.

## 1. Domain-Typen zuerst (`crates/domain`)

1. Modelliere Datenmodelle, Zustände und Fehler-Enums vollständig in `crates/domain` — ohne I/O, ohne FFmpeg-, OS- oder GPU-Abhängigkeiten.
2. **Make Illegal States Unrepresentable**: Ersetze kombinierbare Booleans und `Option`-Ketten durch enge Rust-Enums und validierte Smart-Constructors (z. B. `ExecutionPlan::GpuZeroCopy` akzeptiert niemals CPU-Filter oder Audio).
3. Definiere explizite Fehler-Enums für alle verbotenen Zustände; kein stiller Fallback als Ersatz für einen sauberen Fehlertyp.

## 2. Typentwurf abnehmen lassen

1. Präsentiere dem Nutzer vor dem Schreiben von Boundary- oder Pipeline-Code:
   - die vorgeschlagenen Domain-Typen und Invarianten,
   - welche illegalen Zustände dadurch zur Compile-Zeit ausgeschlossen werden,
   - welche Crate welche Verantwortung übernimmt.
2. Starte erst nach Freigabe des Typentwurfs mit der Umsetzung.

## 3. Systemgrenze isolieren (`crates/probe`)

1. Parse und validiere externe Eingaben (Dateipfade, `ffprobe`-Output, Hardware-Capabilities, Treiber-Flags) exakt einmal in `crates/probe`.
2. Konvertiere Rohdaten an der Grenze direkt in validierte `fastconv-domain`-Typen.
3. Keine wiederholte Defensiv-Validierung in `crates/pipeline`.

## 4. Konkrete Backends ohne spekulative Traits (`crates/pipeline`)

1. Implementiere zuerst konkrete Module (`cpu`, `gpu::nvenc`, `gpu::vaapi`, `gpu::videotoolbox`) mit explizitem `match` auf `ExecutionPlan`.
2. **Laziness Protocol**: Keine Traits, Generics, Factory-Pattern oder Wrapper-Layer für nur einen Aufrufer. Eine Abstraktion entsteht frühestens bei echtem, wiederkehrendem Bedarf mehrerer konkreter Backends.
3. Keine stillen CPU-Fallbacks: Scheitert ein GPU-Pfad, brich mit einem klaren, ursachenbezogenen Fehler ab.

## 5. Dünne Orchestrierung (`crates/cli`) & Abhängigkeits-Check

1. `crates/cli` parst nur den Aufruf, verdrahtet `probe` → `domain` → `pipeline` und mappt Fehler sauber auf Exit-Codes.
2. Prüfe die Crate-Grenzen gegen `AGENTS.md`:
   - `domain`: keine Workspace-Dependencies, zero I/O.
   - `probe` & `pipeline`: nur `fastconv-domain`, niemals gegenseitig.
   - `unsafe_code = "forbid"` bleibt Default; falls FFI/GPU-Treiber später `unsafe` erzwingen, kapsele es minimal mit Safety-Invariante im Modul.
3. Reduziere Leselast: Keine Kommentare, die Code wiederholen — dokumentiere ausschließlich das „Warum“ bei echten Treiber- oder Codec-Quirks.
