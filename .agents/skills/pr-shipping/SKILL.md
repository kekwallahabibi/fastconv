---
name: pr-shipping
description: Verwenden, wenn ein fertig verifiziertes Feature oder ein Bugfix in fastconv als sauberer GitHub Pull Request per gh CLI ausgeliefert werden soll.
---

# PR-Shipping (Adaptierter tstack-Workflow via `gh`)

Liefere Änderungen diszipliniert, nachvollziehbar und mit grünem CI aus.

## 1. Branch- & Repo-Status prüfen

```bash
git branch --show-current
git status --short
```

- Niemals direkt auf `main` committen oder pushen. Falls noch auf `main`, zuerst einen sprechenden Feature-Branch anlegen (`feat/...`, `fix/...`, `perf/...`, `chore/...`).
- Prüfe vor dem Versand mit `adversarial-review`, dass `git diff main...HEAD` frei von Blockern ist.

## 2. Lokale Voll-Verifikation

```bash
cargo xtask verify
```

- Alle Checks (`fmt --check`, `clippy -D warnings`, `test --workspace`) müssen lokal grün sein.
- Halte bei Konvertierungs- oder Performance-Änderungen zusätzlich den Benchmark-/Fixture-Nachweis (`prove-it-works`) bereit.

## 3. Saubere Conventional Commits

Erstelle fokussierte, atomare Commits im Format `<type>(<scope>): <prägnante beschreibung>`:
- Erlaubte Typen: `feat`, `fix`, `perf`, `refactor`, `test`, `chore`, `ci`, `docs`.
- Beispiele: `feat(domain): add zero-copy gpu execution plan`, `fix(probe): reject non-file source paths`.
- Keine unaufgeräumten `wip`- oder `fixup`-Commits im PR.

## 4. Branch pushen & PR anlegen

```bash
git push -u origin "$(git branch --show-current)"
```

Erstelle den PR via `gh pr create` mit kurzem, prägnantem Titel und strukturiertem Body:
- **Was & Warum:** Kurze Zusammenfassung der Änderung und betroffener Crates.
- **Verifikation:** Exakter Nachweis von `cargo xtask verify` (und ggf. Fixture-Formatcheck + gemessene FPS/MB/s vs. Baseline).
- **Risiken / Hardware-Hinweis:** Offenlegen, falls bestimmte GPU-Pfade nur via CI/Unit-Tests verifiziert wurden.

## 5. CI aktiv beobachten & grüne Checks sicherstellen

```bash
gh pr checks --watch
```

- Bei jedem roten Check: Fehlerursache direkt mit `gh run view <run-id> --log-failed` analysieren.
- Root Cause lokal beheben (keine Workarounds, keine deaktivierten Lints), erneut `cargo xtask verify` ausführen, committen, pushen und `gh pr checks --watch` wiederholen, bis alle Checks grün sind.
