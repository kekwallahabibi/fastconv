# fastconv — Engineering-Regeln

`fastconv` ist ein CPU/GPU-Media-Konvertierungstool in Rust. Halte die Architektur flach, explizit und beweisbar korrekt.

## Crate-Routing & erlaubte Abhängigkeiten

| Crate | Pfad | Zuständigkeit | Erlaubte Workspace-Deps |
| :--- | :--- | :--- | :--- |
| `fastconv-domain` | `crates/domain` | Reine Datenmodelle, Invarianten, Fehler-Enums (zero I/O, zero GPU/FFmpeg) | Keine |
| `fastconv-probe` | `crates/probe` | Systemgrenze: externen Input (Dateien, Media-Header, HW-Erkennung) parsen & validieren | `fastconv-domain` |
| `fastconv-pipeline` | `crates/pipeline` | Konkrete CPU- und GPU-Backends für validierte `ExecutionPlan`s | `fastconv-domain` |
| `fastconv-cli` (`fastconv`) | `crates/cli` | Dünne Binary zur Orchestrierung (`probe` → `domain` → `pipeline`) | `fastconv-domain`, `fastconv-probe`, `fastconv-pipeline` |
| `xtask` | `xtask` | Lokale Automation (`cargo xtask verify`, `cargo xtask gen-fixtures`, `cargo xtask bench`) | `fastconv-domain`, `fastconv-probe` |

## Harte Invarianten

1. **`fastconv` ist KEIN FFmpeg-Wrapper**
   In `crates/` darf niemals ein `ffmpeg`- oder `ffprobe`-Subprozess aufgerufen werden. `ffmpeg` darf ausschließlich in `xtask` als externer Benchmark-Gegner (sowie zur synthetischen Video-Fixture-Erzeugung) genutzt werden, um unsere Baseline zu messen, die wir mit nativem Rust + moderner CPU/GPU schlagen wollen.

2. **Model the Domain & Type System Discipline**
   Illegale Zustände (z. B. ein GPU-Zero-Copy-Plan mit CPU-Filter oder GPU-Backend für Audio) müssen über Rust-Enums zur Compile-Zeit unmöglich sein statt Booleans oder `Option`-Ketten zu stapeln.

3. **Boundary Discipline**
   Externer Input (CLI-Argumente, Dateisystem, Media-Header, Treiber-/System-APIs) wird exakt einmal an der Systemgrenze in `crates/probe` geparst und validiert. Ab `crates/domain` existieren nur gültige Typen.

4. **Laziness Protocol & Minimize Reader Load**
   Keine Traits, Wrapper oder Abstraktionsschichten für nur einen Aufrufer. Keine spekulativen Hooks. Keine redundanten Kommentare — kommentiere ausschließlich das „Warum“ bei echten Treiber-, Codec- oder OS-Quirks.

5. **Prove It Works**
   Kein Task ist abgeschlossen, solange `cargo xtask verify` nicht vollständig grün ist. Bei Konvertierungen oder Optimierungen müssen zusätzlich realer Durchsatz (FPS / MB/s), Ziel-Format und der tatsächlich genutzte Hardware-Pfad an `fixtures/` belegt werden.

6. **Fix Root Causes**
   Keine stummen Fallbacks (insbesondere kein stiller CPU-Fallback bei fehlgeschlagenem GPU-Setup), kein Wegfangen von Fehlern mit `let _ = ...` und keine Workarounds, die Treiber- oder Parser-Bugs kaschieren.

7. **Lint-Disziplin**
   Workspace-Lints (`unsafe_code = "forbid"`, `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`, `pedantic = "warn"`) gelten überall. Keine lokalen `#[allow(...)]`-Umgehungen ohne zwingenden, dokumentierten Grund.

## Git-Workflow

Direkte Änderungen und Commits auf `main` sind nicht erlaubt. Halte vor und nach jeder Aufgabe strikt folgenden Ablauf ein:

1. **Branch vorab prüfen:** Prüfe vor jeder Code-Änderung mit `git branch --show-current`, auf welchem Branch du bist.
2. **Branch automatisch anlegen:** Wenn du auf `main` bist, erstelle automatisch zuerst einen passenden Branch (z. B. `git checkout -b feat/<kurzer-name>` oder `git checkout -b fix/<kurzer-name>`), bevor du Dateien bearbeitest.
3. **Verifikation & PR:** Jedes Feature und jeder Fix wird erst nach grüner Verifikation (`cargo xtask verify`) per PR gemerged. Nutze die Projekt-Skills unter `.agents/skills/` (`rust-architecture`, `prove-it-works`, `adversarial-review`, `pr-shipping`).
4. **Nach einem gemergten PR aufräumen:** Wechsle zurück auf `main` (`git checkout main`), ziehe per `git pull` den neuesten Stand und lösche den alten lokalen Branch (`git branch -d <branch-name>`).

> Hinweis: Ein lokaler `.git/hooks/pre-commit`-Hook blockiert zusätzlich versehentliche Commits direkt auf `main`.
