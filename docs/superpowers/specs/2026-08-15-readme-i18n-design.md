# Design: README English / Chinese Split with Language Switch Links

**Date:** 2026-08-15  
**Status:** Approved  
**Goal:** Make every README readable in a single language, with a top-of-page switch between English and Chinese.

## Problem

All project READMEs currently interleave English and Chinese in the same paragraphs, titles, lists, and figure captions (patterns like `Title / 标题`, `**EN**` / `**中文**`, and bilingual fig captions). This is hard to scan for both Chinese and international readers.

GitHub Markdown cannot run interactive JS language toggles, so the practical solution is separate language files with mutual links that act as a switcher.

## Decisions (confirmed)

| Decision | Choice |
|---|---|
| Delivery form | Separate files + top switch links |
| Default file | `README.md` = English |
| Chinese file | `README_zh.md` |
| Figure captions | Language-specific only (no bilingual captions) |
| Scope | All 10 existing README locations |
| Out of scope | Docs site, JS toggles, code/config/asset changes |

## Scope — files to produce

For each path below, keep/create `README.md` (EN) and create `README_zh.md` (ZH):

1. `/` (root)
2. `pre-training/`
3. `classification/linear_eval/`
4. `classification/fewshot/`
5. `detection/`
6. `segmentation/`
7. `visualize/`
8. `dataset/`
9. `weights/`
10. `results/`

Result: 10 English READMEs + 10 Chinese READMEs (20 files total; existing bilingual `README.md` files are rewritten to English-only).

## Language switcher

Place this block near the top of every README (after any centered hero/header on the root README, or as the first content block for simpler READMEs):

```markdown
<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>
```

Chinese file:

```markdown
<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>
```

Rules:

- Current language is bold plain text (not a link).
- Other language is a relative link to the sibling file.
- Same alignment and wording on all files for consistency.

## Content rules

### Split by language

- Titles, section headings, prose paragraphs, bullet lists, table cell descriptions, and figure captions appear in one language only in each file.
- Remove mixed markers such as ` / 中文标题`, `**EN** —`, `**中文** —`, and inline `English; 中文` pairings.

### Keep identical across languages

- Code fences and shell commands (paths, flags, package versions).
- Dataset / model / paper identifiers (e.g. `SARDet-100K`, `iTPN`, arXiv IDs).
- Image paths (`docs/figures/...`) and badge URLs.
- Directory tree layouts (folder/file names stay as in the repo).

### Comments inside code blocks

- English README: prefer English comments in code fences.
- Chinese README: prefer Chinese comments in code fences when the original had Chinese explanation; keep command tokens unchanged.

### Cross-links between directories

- From an English README, link to another directory’s `README.md`.
- From a Chinese README, link to that directory’s `README_zh.md` when it exists (all in-scope directories will have one).

### Figures

- Same image assets in both languages.
- Captions: English-only in `README.md`, Chinese-only in `README_zh.md`.

## Non-goals

- No GitHub Pages / docs theme / real JS toggle.
- No changes to training code, configs, weights, or figure PNGs/PDFs.
- No automatic translation of content that is already present; extract and clean existing bilingual text rather than inventing new technical claims.
- No README outside the 10 listed paths.

## Implementation approach

1. For each existing bilingual `README.md`, produce an English-only `README.md` and a Chinese-only `README_zh.md` by splitting existing content (not rewriting the paper claims).
2. Add the language switcher to every file.
3. Fix internal doc links per language.
4. Spot-check that each pair covers the same sections (structure parity) and that code/commands match.

Recommended process: manual split (content volume is modest; bilingual patterns are inconsistent enough that a brittle extractor script would need more review than it saves).

## Acceptance criteria

- Opening any of the 10 `README.md` files shows English-only prose/captions plus a working link to Chinese.
- Opening any of the 10 `README_zh.md` files shows Chinese-only prose/captions plus a working link to English.
- No mixed `EN / 中文` body text remains in those files.
- Code blocks and resource paths still match the repository layout.
- No unrelated code or asset changes.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Drift between EN and ZH over time | Keep section structure parallel; future edits update both files |
| Broken relative links | Use `./README.md` / `./README_zh.md`; verify subdirectory cross-links |
| Accidental content loss while splitting | Diff section headings EN vs ZH before finishing each pair |
