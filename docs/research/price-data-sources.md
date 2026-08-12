# Polish Used-Car Price Data — What Exists, and Under What Licence

Date: 2026-08-12
Status: accepted
Author: P0w3r223 + Claude
Related to: [ADR 0001](../adr/0001_scope-and-site-expansion.md) (W3, W4),
[data-and-methodology.md](data-and-methodology.md)

---

Research question: for a publicly published portfolio project that cannot scrape listing
portals, what Polish used-car price data exists today — how current, how large, how
granular, and under what licence?

## The headline corrections to this project's own documentation

**The training data is a single January 2022 snapshot, not "~2021–2023".** The claim
appears in the README, in `data-and-methodology.md` and in the `config.py` module
docstring. Kaggle's metadata for `aleksandrglotov/car-prices-poland` records a January 2022
Selenium scrape, and the model-year distribution corroborates it independently: 2021
accounts for 8.96 % of rows, 2022 for 1.78 %, and nothing follows. A dataset genuinely
spanning 2021–2023 would contain 2023 model years. *(Verified locally against the raw CSV.)*

Consequences: `REFERENCE_YEAR = 2024` sits two years past the data vintage, and the model
has never seen a car younger than `age` 2 while the API accepts model year 2024.

**A CC0 tag from a scraper is worth less than it looks.** Every open Polish car dataset was
uploaded by the person who scraped it, not by the database maker. A CC0 or MIT tag granted
by someone without title cannot extinguish the portal's *sui generis* database right. This
applies to the dataset this project already uses. The defensible position is "we do not
**republish** third-party database extracts", not "the tag makes it clean" — and the
provenance chain, not just the tag, belongs in the attribution.

**The legal rationale in `data-and-methodology.md` is out of date.** Poland implemented the
DSM Directive by the Act of 26 July 2024 (in force 20 September 2024), amending the
Copyright Act *and* the Database Protection Act. Disseminated databases may now be
reproduced for text and data mining **unless the maker reserves otherwise**, and for online
databases that reservation must be machine-readable. Otomoto's `robots.txt` currently emits
`Content-Signal: ai-train=yes, search=yes, ai-input=yes` — signalling permission, not
reservation, and the opposite of Cloudflare's managed default.

Three things still bite, so narrow the conclusion rather than reverse it: Content Signals
address AI training rather than a general Art. 4 TDM reservation; the site's *Regulamin* is
a separate contractual layer (*Ryanair v PR Aviation*, C-30/14, holds that the maker of an
unprotected database may restrict use contractually); and decisively, the TDM exception
covers reproduction **for mining**, never **re-utilisation** — publishing a 100k-row extract
in a public repository is a separate restricted act.

## No Polish government source carries a price

Verified by direct query, not assumption. A CEPiK vehicle record contains exactly: `marka`,
`kategoria-pojazdu`, `typ`, `model`, `wariant`, `rodzaj-pojazdu`, `pochodzenie-pojazdu`,
`rok-produkcji`, `data-pierwszej-rejestracji-w-kraju`, `pojemnosc-skokowa-silnika`,
`masa-wlasna`, `rodzaj-paliwa`, `wojewodztwo-kod`. No price, value, `cena` or `wartość`
field. A 20-result sweep of dane.gov.pl for "samochody" returns registration counts, theft
rates, cars per capita, border traffic — zero price datasets. GUS publishes CPI indices, not
average used-car prices; the "average price" figures in the Polish press come from AAA AUTO.

**This is a positive finding in disguise.** CEPiK is free, weekly-updated, commercial reuse
permitted with attribution, and it is the *national population* over exactly the covariates
this model uses. It cannot supply `y`, but it can validate and reweight `X` — and the model
currently trains on one portal's convenience sample with no check that its make/age/fuel mix
resembles Poland's actual fleet.

## Open microdata: all Otomoto scrapes, nothing genuinely new since August 2023

| Dataset | Collected | Rows | Licence | Notes |
|---|---|---|---|---|
| `bartoszpieniak/poland-cars-for-sale-dataset` | May 2021 | 208 304 | CC0 | 25 cols incl. `Power_HP`, `CO2_emissions`, `Drive`, `Transmission`, `Origin_country`, `First_owner` |
| `aleksandrglotov/car-prices-poland` | **Jan 2022** | ~117k | CC0 | current source |
| `krzysztofdogowski/used-cars-poland-with-links-may-2022` | May 2022 | 140k | CC0 | includes advert links |
| `wspirat/poland-used-cars-offers` | Jun 2023 | ~90k | CC0 | |
| `szymoncyperski/car-sales-offers-from-otomotopl-2023` | **Apr 2023** | >200k | CC BY-SA 4.0 | all offers posted April 2023 |
| `krzysztofdogowski/used-cars-in-poland-many-parameters` | Aug 2023 | ~200k | **Unknown** | richest, but unlicensed — do not use |

Everything dated 2024–2026 is a re-upload: `nakib92/car-price-poland` is byte-for-byte the
size of the aleksandrglotov file; `anshid170/poland-cars-sales-data` describes itself as the
bartoszpieniak set. HuggingFace returns nothing for `otomoto` or `car price poland`.

## Platform APIs are closed — all three routes are dead ends

Otomoto's API is a write/management API for business customers to publish their own
listings; there is no market-read endpoint. Allegro's `GET /offers/listing` has been closed
to new applications since March 2021. DSA Art. 40 vetted-researcher access (delegated act
adopted 2 July 2025) applies only to designated VLOPs — Allegro, OLX and Otomoto are not
designated, and the route additionally requires a research-organisation affiliation.

## Public-sector auction data: legally cleanest, practically useless

eLicytacje KAS (live since 1 July 2026) and `licytacje.komornik.pl` publish *suma
oszacowania* and *cena wywołania* for vehicles. Open Data Directive Art. 1(6) bars public
sector bodies from invoking the sui generis right against re-use, which is a materially
stronger position than any Kaggle CC0 tag. But forced-sale prices sit far below market (a
sampled notice: Honda Civic, appraisal 4 000 PLN, opening 2 000 PLN), volume is low,
attributes are unstructured free text, there is no bulk export, and notices carry
registration plates. Worth a documented negative result, not training data.

## Free aggregate benchmarks — no microdata, but enough to sanity-check a model

- **AAA AUTO Barometr**, monthly: average price of cars *actually sold* — rare, as
  everything else is asking price — nationally and for all 16 voivodeships. April 2026:
  mazowieckie 72 659 PLN against warmińsko-mazurskie 45 171 PLN, a 27k spread. **Directly
  comparable to this project's `province` feature and the best available external check on
  its regional effects.**
- **OTOMOTO Insights**, monthly: national median, segment averages (Q1 2026: city ~17k,
  compact ~44k, SUV ~88k PLN), average listing lifetime.
- **ZDS + Autoplac**, quarterly, authorised-dealer segment. **IBRM SAMAR**: volumes free,
  price reports paywalled.

## Commercial options

Autovista/Eurotax (JD Power) and Info-Ekspert are enterprise, price on application, no
public academic tier found. **CarDossier** is the closest off-the-shelf reference — 618
models across 63 brands from 1.5M listings, current to August 2026, with a `/regional`
endpoint over the 16 voivodeships and a free 50-credit trial — but it is itself an
aggregation of scraped listings, so the upstream legal chain is the one this project
rejected, and its redistribution terms are unpublished. **Bright Data does not cover
Poland.** Buying from a scraping vendor does not launder the sui generis right; the
purchaser becomes a re-utiliser.

## One non-Polish academic dataset with a real, grantable licence

**DVM-CAR 2.0**: 335 562 UK used-car adverts, 899 models, 1.45M images, sales data sourced
from the DVLA, licensed **CC BY-NC** (DOI 10.6084/m9.figshare.19586296). Wrong market and
non-commercial only, but the licence chain is sound and it is citable — usable as a second
market to show the pipeline generalises.

## Actionable shortlist

1. Correct the vintage claim in the README, this directory's methodology doc and
   `config.py`; revisit `REFERENCE_YEAR` against it.
2. Add the **April 2023 snapshot (CC BY-SA 4.0)** as a second source — fifteen months newer,
   turning staleness into a measured temporal-drift demonstration. Share-alike propagates to
   derived datasets.
3. **Do not** use `used-cars-in-poland-many-parameters` — its licence is literally "Unknown".
4. Use **CEPiK** for covariate validation and reweighting against the national fleet.
5. If a time-adjustment layer is added, use Eurostat PL/CP07112 but document that it is
   quality-adjusted and diverges ~7 pp from raw market medians — the two answer different
   questions. See [temporal-recalibration.md](temporal-recalibration.md).
6. Use AAA AUTO's voivodeship table as an external check on the model's province effects.

## Gaps and uncertainties

- **Eurostat continuity:** `prc_hicp_midx` is flagged discontinued in favour of
  `prc_hicp_minr`, but queries to the replacement returned HTTP 400 across several dimension
  combinations. PL/CP07112 is confirmed only through 2025-12. Verify before wiring anything.
- **Otomoto's Regulamin was not read** — only `robots.txt`. The contractual layer is
  asserted from secondary Polish legal commentary, so the Content Signals finding is weaker
  than it appears until the terms are checked.
- **CarDossier redistribution terms** are not published; do not assume the trial permits
  publishing derived figures.
- `belzebbdbo/otomotov2-dataset` (MIT, June 2025, 16.5 GB) is completely undocumented and is
  the only 2025 Otomoto artefact that is not obviously a re-upload. Whether it holds tabular
  prices needs a manual download.
- **No transaction-price microdata exists in any source found.** Every microdata source
  carries asking prices; the only transaction signal is AAA AUTO's monthly aggregate.
- Dealer-published inventory feeds (the XML/CSV export format used for portal integrations)
  were not investigated. Data published by its own owner would be a legally clean route, if
  it exists at usable scale.

## Sources

CEPiK API (`api.cepik.gov.pl/pojazdy`) and its gov.pl terms · dane.gov.pl datasets API ·
Open Data Directive Art. 1(6) · Eurostat `prc_hicp_midx` / `prc_hicp_manr` (PL, CP07112) ·
Kaggle datasets API (metadata for the six datasets above) · HuggingFace datasets API ·
DVM-CAR 2.0 (deepvisualmarketing.github.io, arXiv 2109.00881) · Allegro API issue #4213 ·
Otomoto API FAQ · DSA Art. 40 delegated act (2 July 2025) and VLOP designation lists ·
Autovista/Eurotax, Info-Ekspert, CarDossier, Bright Data product pages · OTOMOTO Insights ·
AAA AUTO Barometr via SAMAR · ZDS/Autoplac Q1 2026 · Act of 26 July 2024 (Sejm druk 406) and
consolidated Database Protection Act (Dz.U. 2024 poz. 1769) · Otomoto `robots.txt` ·
Cloudflare Content Signals Policy · eLicytacje KAS · licytacje.komornik.pl category 24
