---
title: "Pale Lights — Seed → Generation Comparison"
---

# Pale Lights — Seed → Generation Comparison

**Date:** 2026-04-14
**Seed:** `seeds/pale_lights.json` (hand-authored from `rollup.gemma4.md`; 48 entities, 10 hooks, 4 motifs, 6 threads, 7 rules, 3 themes, full narrator config)
**Model:** Gemma 4 26B A4B (PRISM-PRO-DQ quant) via `llama-server` at 127.0.0.1:8082
**Pipeline:** arc → dramatic → emotional → craft → per-beat writer loop (17 beats) → craft-critics → check → revise → check → extract
**Action:** *"Tristan takes the cabinet job Abuela set him — slipping into the Azulejo hostel in the Estebra District at third watch to lift a locked cabinet from a stranger's room."*
**Output:** `/tmp/pl_quest/prose.txt`, 2,389 words, 16m30s wall-clock, outcome=committed
**Reference:** Pale Lights Ch 1, 5,438 words (same opening action).

## Method

Authored `seeds/pale_lights.json` as a "Doyle-view" omniscient start state drawn from the rollup. The seed includes 16 named characters (2 ACTIVE protagonists + 2 ACTIVE pact-gods + 12 DORMANT), 6 locations, 9 factions and concepts (the Watch, Hoja Roja, Red Eye, Law of Rats, trials, Iscariot Accords, Signs, Red Maw truth), 7 items (Spinster's Milk, lodestone extract, the List, the Rhadamanthine pistol, Angharad's saber, Osian's letter, the stolen cabinet papers), 6 plot threads (Tristan's List, Angharad's revenge, the three trials, Hoja Roja hunts Tristan, Red Maw containment, Yong's debt), 10 pre-planted foreshadowing hooks, 4 motifs with semantic ranges, 3 themes, and a narrator block with `pov_type: third_limited_alternating`, voice samples, register notes, sensory bias, and attention bias.

Ran `quest init` + direct-Python pipeline invocation (had to bump `InferenceClient.timeout` from 120s to 600s — default blew up on a 16k-token check-stage prompt).

## Headline result

**Seed density is not the bottleneck. The writer and narrator plumbing are.**

The dramatic planner used seeded content well: scene 1 correctly names Azulejo hostel, Estebra District, Fortuna, Abuela, the cabinet, the Guardia, "ticking in his head"; scene 3 switches POV to Angharad and names the Fisher, the Bluebell, the Guardia pursuit. POV alternation emerged without dedicated plumbing. The prior "thin world" failure was accurately diagnosed as a seed problem — with 48 seeded entities and full narrator config, the planner reached for them.

But two downstream failures turned a good plan into a weak chapter:

## Failures

### 1. Writer renders Fortuna as Tristan's cat

The dramatic plan correctly listed `characters_present: ["tristan", "fortuna"]` with the beat *"Fortuna sits on the edge of the bed, watching the shadows with golden, bored eyes."*

The writer produced:

> Fortuna sat perched on the edge of the bed, a small, silent weight against the linen. Her eyes were two molten coins of amber, burning through the gloom. She watched the shadows. They drifted, long and spindly, across the floor, but her gaze remained heavy with a profound, ancient boredom. **She did not stir. She did not meow.** She was simply there—a golden-eyed witness to the dark, her ears twitching once at a sound you had yet to hear.

Fortuna — a girl-shaped goddess in a red dress with hair and eyes of molten gold, invisible to everyone but Tristan — was rendered as a cat. The seed describes her across ~200 words (8 distinct traits: girl-shape, red dress, gold hair/eyes, bellows, pouts, rests chin on Tristan's shoulder, feuds with inanimate objects, collection of terrible habits made into a deity). The writer used exactly one trait (gold eyes), inferred "small, silent weight against the linen", and generated a cat.

**This is the core fidelity failure.** Named entities from the dramatic plan are not being bound to their full seed descriptions when the writer sees them. Either (a) the entity-lookup into writer context only supplies name + role, not the seed `data` blob, or (b) the writer prompt does not instruct the model to treat named entities as authoritative seeded definitions.

### 2. Narrator POV config ignored

Seed specifies `pov_type: third_limited_alternating` with all five voice samples in third person. The writer produced second-person CYOA:

> You slipped through the service entrance, moving with the liquid grace of a shadow cast by a dying candle.

Compare Erratic's Ch 1:

> None of the skeleton keys were working. The landlord must have sprung for good locks, which was admittedly sensible of the man considering that Tristan was currently trying to rob one of his patrons.

The `Narrator` config is not reaching the writer's system prompt (or there is a hard default in `prompts/stages/write/system.j2` that overrides). This has nothing to do with seed density — it's a pipeline wiring bug.

### 3. Plot drift: the cabinet contents

Seed (and canon): the cabinet is a poisoner's kit with drawers of *black verity* and vials of *Spinster's Milk*, and the papers reveal the stranger was Hoja Roja's man. Cabinet contents and poisons are seeded as `item:spinsters_milk`, `item:cabinet_papers`, plot thread `pt:hoja_roja_hunts_tristan`.

The dramatic planner invented a different plot: the papers are *shipping manifests* for a cargo-carrying Bluebell, and the Bluebell is "the heart of a web" of trade routes. The seeded poisons never appear. Hoja Roja never appears. The Law of Rats (seeded as motif + concept with its full verse) is never invoked.

The planner used seeded *names* but drifted on specific *plot content*. This is a lower-severity failure than the Fortuna-as-cat one, but it shows the dramatic planner is not consulting item `data` fields or plot_thread descriptions closely enough to stay on-canon. The concept entity `concept:red_maw_truth` and plot thread `pt:red_maw_containment` (both explicitly seeded) were not touched.

### 4. The stranger Tristan kills is missing

Ch 1's central action is Tristan killing the Watchman Yaotl Cuatzo. My seed did not include Yaotl as an entity — seed-side gap, my oversight. With no Yaotl in the entity list, the dramatic planner had no one to populate the room, and scene 1's beat sequence ends with *"The lock clicks; the cabinet is open"* rather than a confrontation. The Guardia pass in the hall but no one is actually in the room.

**Seed lesson:** opening-state seeds need the inciting-scene NPCs even if they are "about to die" — a one-line DORMANT entity would have let the planner stage the fight.

### 5. Second POV truncates mid-sentence

Scene 3 (Angharad on the docks) starts at word ~1,800 and cuts off at word 2,389 with *"The air was thick, a heavy"* — the writer loop stopped mid-beat. Likely the per-beat budget ran out, or a beat-level call errored silently and the pipeline kept going to extract. Tristan got 17 writer calls with full narrative; Angharad got partial coverage of one scene.

### 6. Zero use of seeded texture

Named things seeded and never referenced in output: Law of Rats, Spinster's Milk, lodestone extract, black verity, Hoja Roja, the Watch (by name), blackcloaks, Gongmin lock, Rhadamanthine quartz, Tristan's List, the Fisher, Iscariot Accords, Manes, Red Maw, Cozme Aflor, Gloam, the Glare. These are the *texture* Erratic reaches for — 11 new named facts per chapter. The seed had all of them; the writer used almost none.

## What worked

- POV alternation across scenes without explicit plumbing.
- Correct location naming (Azulejo hostel, Estebra, Murk, docks).
- Fortuna present in plan (even if garbled in render).
- Pipeline completed end-to-end: arc/dramatic/emotional/craft planners + 17 writer beats + check/revise + extract, all stages produced committed output.
- Atmospheric writing quality at the sentence level is decent; the prose itself is not bad, it's just not Pale Lights and not in the voice the seed asked for.

## Diagnosis — which layers need work

Prior plan (in order): (1) hand-authored seed, (2) seed-density validation, (3) DORMANT→context wiring, (4) `entities_to_surface` on DramaticPlan, (5) seed generator, (6) defer runtime world-fact generation.

**Empirical revision based on this run:**

| # | Original | Empirical priority |
|---|---|---|
| 1 | Rich seed | ✓ Done; density works. Add missing inciting-scene NPCs to the Pale Lights seed (Yaotl). |
| — | *(new)* | **Writer entity-fidelity.** When the writer prompt mentions a named entity, the full seed `data` blob for that entity must be in-context, and the prompt must instruct the writer to treat it as authoritative. This is the Fortuna-as-cat fix. Highest leverage of anything on this list. |
| — | *(new)* | **Narrator config reaches the writer.** The `Narrator` from seed must pin the writer's POV/register/voice-samples. The writer is currently defaulting to CYOA 2nd person regardless of seed. |
| 2 | Seed-density validator | Deprioritized — not the bottleneck. Still worth doing as a cheap warning. |
| 3 | DORMANT in context | Worth doing; DORMANT entities (Cozme, Hoja Roja, Law of Rats, poisons) would give the dramatic planner more to pull from and would likely have produced the real Ch 1 fight. |
| 4 | `entities_to_surface` | Natural follow to #3. |
| — | *(new)* | **Plot-content fidelity on the dramatic planner.** Seeded `plot_threads` and `items` have `description` and `data` fields; the planner is reading names but drifting on specifics (cabinet = manifests instead of poisons). Worth a prompt revision that quotes the seeded descriptions of items/threads referenced in a scene. |
| — | *(new)* | **Writer loop budget/stop.** Second POV truncated mid-beat; investigate whether per-update token budget is getting spent on the first POV. |
| 5 | Seed generator | Still deferred. |
| 6 | Runtime world-fact generation | Still deferred — if anything this run strengthens the defer: when the planner *did* invent ("manifests", "cargo-carrying Bluebell"), the inventions conflicted with seeded canon. |

## Bottom line

The bet that "rich seed + dormant activation → Pale Lights density" is half-right. The seed side of the bet is correct: 48 entities, 10 hooks, full narrator is enough content for the planner to pull from, and it did. The activation side is wrong: the *writer* doesn't honor the seed. Fortuna becomes a cat because the writer sees "Fortuna, on the bed, golden eyes" and free-associates without grounding. The narrator becomes 2nd-person CYOA because the seed's Narrator block is not reaching the writer's system prompt.

The next fix is not more seed, not validation, not DORMANT wiring. It is: **make the writer read the seed.** Pipe the full entity `data` blob for every character referenced in a beat into the writer's context, and make the Narrator config the writer's system prompt. Everything else on the prior plan comes after that.

## Artifacts

| Path | Content |
|---|---|
| `seeds/pale_lights.json` | Hand-authored Pale Lights Book 1 start-state seed (48 entities, full narrator) |
| `/tmp/pl_quest/prose.txt` | Generated chapter, 2,389 words |
| `/tmp/pl_quest/traces/6c5fba34…json` | Full pipeline trace with per-stage prompts + outputs |
| `/tmp/pl_quest/pale_lights.db` | Post-run SQLite world state |
