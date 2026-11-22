---
name: adversarial-review
description: Verwenden vor dem Erstellen eines Pull Requests oder wenn der Nutzer ein kritisches Code-Review der aktuellen Branch-Änderungen in fastconv anfordert.
---

# Adversarial Review (Feindliches Diff-Audit)

Betrachte den aktuellen Diff mit maximaler Skepsis. Ziel ist es, echte Defekte, Regelverstöße und ungeprüfte Annahmen zu finden — nicht nett zu sein.

## 1. Exakten Diff laden

Prüfe ausschließlich die realen Änderungen gegenüber `main`:

```bash
git status --short
git log main..HEAD --oneline
git diff main...HEAD
```

Lies bei jeder geänderten Funktion genug Kontext, um Aufrufer und Invarianten sicher zu beurteilen.

## 2. Feindliche Prüfliste

Suche gezielt nach jedem der folgenden Fehlerklassen:

1. **Versteckte Panics & Lint-Umgehungen**
   - `unwrap()`, `expect()`, `panic!()`, `todo!()`, `unimplemented!()`, `unreachable!()`
   - Ungeschützte Indexzugriffe (`slice[i]`), arithmetische Overflows oder Cast-Truncation (`as u32` / `as usize`)
   - Lokale `#[allow(...)]`-Attribute, die Workspace-Lints aushebeln
2. **Stilles Fehler-Verschlucken & Fallbacks**
   - `let _ = ...`, `.ok()`, ignorierte `Result`s, zu breite `Err(_)`-Matches
   - Stille CPU-Fallbacks bei fehlgeschlagenem GPU-Backend oder Treiber-Init
3. **Unnötige Abstraktionen (Laziness-Verstoß)**
   - Traits mit nur einem Implementierer, Pass-through-Wrapper, tote Hilfsfunktionen, spekulative Konfig-Flags
   - Redundante Kommentare ohne „Warum“-Mehrwert
4. **Kaputte Boundary-Checks & Architektur-Lecks**
   - Ungeprüfter externer Input außerhalb von `crates/probe`
   - I/O, FFmpeg- oder GPU-Aufrufe in `crates/domain`
   - Verbotene Crate-Dependencies gegen die Routing-Tabelle in `AGENTS.md`
   - Illegale Zustände, die über Booleans statt Enums darstellbar bleiben
5. **Fehlende Verhaltenstests & ungeprüfte Annahmen**
   - Fehlende Tests für Fehlerpfade und Edge-Cases (leere Datei, korrupte Metadaten, identischer Zielpfad, fehlende GPU)
   - Schein-Tests, die nur Triviales prüfen oder reale Fehler maskieren
   - Unbelegte Performance- oder Hardware-Behauptungen

## 3. Ausgabeformat (streng faktenbasiert)

Gib für jedes Problem ein konkretes Finding aus:

- **`[Blocker | Warnung | Nit]` `pfad/zur/datei.rs:ZEILE`**
  - **Problem:** Was genau bricht oder gegen welche Invariante verstoßen wird.
  - **Fix:** Konkreter, minimaler Korrekturvorschlag.

Wenn keine Blocker vorliegen, benenne explizit:
1. alle geprüften Dateien/Bereiche,
2. verbleibende Restrisiken oder nicht lokal testbare Hardware-Pfade (z. B. fehlende GPU).
