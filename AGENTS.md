# fastconv — Engineering-Regeln

`fastconv` ist ein CPU/GPU-Media-Konvertierungstool in Rust. Halte die Architektur flach, explizit und beweisbar korrekt.

## Crate-Routing & erlaubte Abhängigkeiten

| Crate | Pfad | Zuständigkeit | Erlaubte Workspace-Deps |
| :--- | :--- | :--- | :--- |
| `fastconv-domain` | `crates/domain` | Reine Datenmodelle, Invarianten, Fehler-Enums (zero I/O, zero GPU/FFmpeg) | Keine |
| `fastconv-probe` | `crates/probe` | Systemgrenze: externen Input (Dateien, Probe-Output, HW-Erkennung) parsen & validieren | `fastconv-domain` |
| `fastconv-pipeline` | `crates/pipeline` | Konkrete CPU- und GPU-Backends für validierte `ExecutionPlan`s | `fastconv-domain` |
| `fastconv-cli` (`fastconv`) | `crates/cli` | Dünne Binary zur Orchestrierung (`probe` → `domain` → `pipeline`) | `fastconv-domain`, `fastconv-probe`, `fastconv-pipeline` |
| `xtask` | `xtask` | Lokale Automation (`cargo xtask verify`, `cargo xtask bench`) | Keine |

## Harte Invarianten

1. **Model the Domain & Type System Discipline**
   Illegale Zustände (z. B. ein GPU-Zero-Copy-Plan mit CPU-Filter oder GPU-Backend für Audio) müssen über Rust-Enums zur Compile-Zeit unmöglich sein statt Booleans oder `Option`-Ketten zu stapeln.

2. **Boundary Discipline**
   Externer Input (CLI-Argumente, Dateisystem, `ffprobe`-Output, Treiber-Flags) wird exakt einmal an der Systemgrenze in `crates/probe` geparst und validiert. Ab `crates/domain` existieren nur gültige Typen.

3. **Laziness Protocol & Minimize Reader Load**
   Keine Traits, Wrapper oder Abstraktionsschichten für nur einen Aufrufer. Keine spekulativen Hooks. Keine redundanten Kommentare — kommentiere ausschließlich das „Warum“ bei echten Treiber-, Codec- oder OS-Quirks.

4. **Prove It Works**
   Kein Task ist abgeschlossen, solange `cargo xtask verify` nicht vollständig grün ist. Bei Konvertierungen oder Optimierungen müssen zusätzlich realer Durchsatz (FPS / MB/s), Ziel-Format und der tatsächlich genutzte Hardware-Pfad an `fixtures/` belegt werden.

5. **Fix Root Causes**
   Keine stummen Fallbacks (insbesondere kein stiller CPU-Fallback bei fehlgeschlagenem GPU-Setup), kein Wegfangen von Fehlern mit `let _ = ...` und keine Workarounds, die Treiber- oder Parser-Bugs kaschieren.

6. **Lint-Disziplin**
   Workspace-Lints (`unsafe_code = "forbid"`, `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`, `pedantic = "warn"`) gelten überall. Keine lokalen `#[allow(...)]`-Umgehungen ohne zwingenden, dokumentierten Grund.

## Git-Workflow

- Niemals direkt auf `main` entwickeln.
- Jedes Feature und jeder Fix entsteht auf einem eigenen Branch und wird erst nach grüner Verifikation (`cargo xtask verify`) per PR gemerged.
- Nutze die Projekt-Skills unter `.agents/skills/` (`rust-architecture`, `prove-it-works`, `adversarial-review`, `pr-shipping`).
