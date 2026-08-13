# ADR 0003 — The valuation form is generated from the config, and refuses to run without it

Date: 2026-08-13
Status: accepted
Author: P0w3r223
Related to: [ADR 0002](0002_site-quotes-the-code.md) (the same discipline, applied to the
other half of the site), [ADR 0001](0001_scope-and-site-expansion.md) (W6)

---

## Context

ADR 0002 rebuilt the report page so that it cannot disagree with the model: every number is
measured, every rule quoted from its module, and CI rebuilds the page and fails on any diff.
The valuation form under `docs/app/` was left exactly as v0.1 wrote it, which made it the
place where every property that page now guarantees was false.

Three problems, each with a history rather than a hypothesis behind it:

- **The form restated what `config.py` owns.** Six numeric bounds and two vocabularies were
  spelled a second time, in JavaScript. One of those copies drifted by a single capital
  letter: the form sent `Kujawsko-Pomorskie` where the model was trained on
  `Kujawsko-pomorskie`. A one-hot encoder answers a near-miss with an all-zero row rather
  than an error, so around 7 % of the market was priced as if the car had no location at all
  — measured at 1 376 PLN mean deviation, p95 6 344, max 42 533 — and returned with a 200.
  The fix at the time was a test asserting the two lists were identical, which pins a copy
  without removing it.
- **The reader met the price before the warning.** When no API answered, the form fell back
  to an offline heuristic and said so *in the result*, under the number, distinguished from a
  real prediction by the colour of a badge. The published demo on GitHub Pages is always in
  that state, so its default experience was: enter a car, receive a large confident figure,
  then read that it is not from the model.
- **It looked like a different project.** Its own stylesheet, its own palette, no dark mode
  and a Google Fonts request, reached through the report's own "Try the valuation form" link.

The first of these is the project's governing rule turned inside out. Everywhere else, an
input the pipeline cannot honestly price is refused; here, a value the pipeline could not
price was manufactured by the form itself.

## Decision

### 1. Everything the form validates against is generated from `config.py`

`site/form.py` writes `docs/app/config.json` — the two vocabularies, the year window, the
mileage and displacement ceilings, the field-length limits, and the one fuel for which zero
displacement is a fact. `app.js` restates none of them. Two tests enforce the absence rather
than the agreement: no province name appears in its code, and no integer literal in it equals
a bound `config.py` owns.

Three constants moved into `config.py` to make this possible — `ELECTRIC_FUEL`,
`MARK_MAX_LENGTH`, `MODEL_MAX_LENGTH` — so the displacement rule and the field limits are
stated once for `data.py`, the API and the form.

### 2. Without that file, the form refuses to run

Missing, unreachable, not JSON, stamped with a different `schema`, or missing any single key:
the form renders "This form cannot run", leaves the submit button disabled and stops. There
are no fallback constants to fall back to, deliberately — a form validating against its own
guess of the domain looks exactly like a working one, which is how the province bug survived.

The button ships `disabled` in the markup and is enabled only after the config has loaded
*and* the backend has been probed, so a failure before JavaScript ever runs fails the same
way. (A disabled default submit button also suppresses Enter-key submission, so the gate is
not merely click-shut.)

The same rule reaches the offline heuristic: a fuel outside its factor table is refused
rather than priced as petrol. Its table is the one place fuel names still appear in
JavaScript, because the factors are the heuristic's own invention rather than a mirror of the
domain — but a domain value falling through to a default is precisely the shape this project
keeps finding, so it raises, and a test additionally keeps the table covering `KNOWN_FUELS`.

### 3. What will answer the form is stated above it, before a price exists

`GET /health` is probed at load — bounded at 3 s, so a host that accepts the connection and
never replies cannot hold the form shut — and the banner above the form reports one of the
three things that can be true: a trained model is answering, the service is up but has no
model loaded, or no API is reachable and this is the static demo.

A 200 that is not this API's health payload counts as "no service", not as "no model": a
static host answering `/health` with a page must not be reported as a service that merely
lacks an artifact.

The answer is then typed as well as labelled. A prediction is a solid-bordered card with the
exact figure and the vintage it is a valuation of; a heuristic answer is a dashed-bordered
card with a lighter, approximate figure (`≈`). The two used to differ by a badge colour and
nothing else, which let the fallback read as a valuation.

### 4. One palette, generated with the page

The stylesheet splits into `tokens.css` (the palette, both schemes) and `base.css` (page
frame and typography), then `report.css` / `form.css`. `site.build` writes `docs/index.html`
and the form's `styles.css` and `config.json` in one command, and CI diffs all three — so a
form validating against last month's vocabulary fails in CI instead of going live.

`--positive` was darkened from `#059669` to `#047857` in the process: 3.5:1 on `--surface` is
below AA for text that size, and it is the label on the one status a reader has to be able to
trust.

## Options considered

1. **Keep the copies, keep the parity tests.** Cheapest, and it is what v0.2 already did. The
   tests do catch drift — but only for values someone remembered to pin, which is the same
   dependency on memory that produced the bug.
2. **Generate a `config.js` assigning a global,** loaded by a `<script>` before `app.js`.
   Synchronous, works over `file://`, no fetch failure mode. Rejected narrowly: the payload is
   data, and a JSON file is what the parity test can compare directly against
   `config_payload()` without parsing JavaScript.
3. **Emit into `docs/data/` beside the report's aggregates.** Rejected on a hard constraint —
   the API mounts `docs/app` at its root, so `../data/config.json` resolves only in the Pages
   layout. The form would have loaded its vocabulary in one deployment and refused to start in
   the other.
4. **Generate `docs/app/index.html` too, from a template.** Rejected as unearned: its content
   is not derived from any measurement, and generating it would add a build step without
   adding a guarantee.
5. **Generated `config.json`, fetched, fail-closed (chosen).**

## Consequences

- **`docs/app/` now needs `config.json` beside it.** Serving that directory without it yields
  a form that refuses to run — loudly, which is the intent, but it is a deployment
  requirement that did not exist before.
- **Editing the form's styling means editing `src/car_price_ml/site/assets/`,** not
  `docs/app/styles.css`; the latter is generated and CI overwrites-and-diffs it.
- **`build` writes three files, not one,** so the page and the form can never be published
  from two different commits of the same constants.
- **The published page carries `--danger`,** a token it does not use, as the cost of one
  shared palette.
- **122 → 131 tests.** The two parity tests that pinned the copies are replaced by tests that
  assert no copy exists.

### Accepted residual risk

No test executes `app.js`. There is no JavaScript runtime or browser in the environment this
was built in, so the fail-closed path, the three banner states and the dark-mode rendering
are verified by reading and by structural assertions on the served markup — not by running
them. The Python side of the contract is pinned (the committed `config.json` matches
`config_payload()`, and `CONFIG_SCHEMA` in `app.js` matches `FORM_CONFIG_SCHEMA`), which is
the half a test can reach without a browser. Closing the other half needs a headless-browser
test, which would be this repository's first Node dependency; it is deliberately left open
rather than approximated.
