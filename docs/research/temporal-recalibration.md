# Recalibrating a 2022-Vintage Price Model to Present-Day Money

Date: 2026-08-12
Status: accepted
Author: P0w3r223 + Claude
Related to: [ADR 0001](../adr/0001_scope-and-site-expansion.md) (W4),
[price-data-sources.md](price-data-sources.md)

---

Research question: which published price series can convert this model's output into
present-day money, how are they obtained programmatically, and what does documented practice
say about the failure modes — in particular, does a single scalar survive segment-specific
drift?

**Answer in one line: it does not, and the obvious adjustment has the wrong sign.** This
research reversed the freshness decision in ADR 0001.

## 1. A general-inflation adjustment would be wrong by 50–65 percentage points

Three series over the same window (Eurostat figures verified by live API query):

| Series | Jul 2022 | Latest | Change |
|---|---:|---:|---:|
| PL all-items HICP (CP00) | 109.5 | 154.7 (Dec 2025) | **+41 %** |
| PL **second-hand cars** (CP07112) | 81.7 | ~62.4 (Jun 2026) | **−24 %** |
| Indicata PL retail price index (Jan 2020 = 100) | ~120 | 108.5 (Oct 2025) | **≈ −10 %** |

Deflating 2022 prices by consumer inflation gives roughly **+43 %**; both car-specific
series give **−10 % to −24 %**. The naive move is not merely imprecise — it is directionally
wrong.

The mechanism is documented: Poland registered 967 579 imported used cars in 2024, +20 %
year on year, and Autovista/JD Power attribute the value slide to oversupply plus a EUR/PLN
rate that killed re-export demand. Indicata recorded Poland at 93.8 in March 2025 — the
steepest decline among large EU markets.

## 2. The two car-specific series disagree by ~14 pp, and one is suspect

Eurostat CP07112, 2015 = 100, January values:

| Country | Jan 2015 | Latest |
|---|---:|---:|
| Germany | 98.5 | **159.7** (Jan 2026) |
| Czechia | 99.4 | **137.1** (Dec 2025) |
| Euro area | 100 | ~137 (2025 avg) |
| **Poland** | ~100 | **~62** (Jun 2026) |

Poland is a sign-flip outlier against every neighbour, over a decade, in a market importing
~500k cars a year *from Germany*. **Inference, not a documented finding:** a sustained 100 pp
divergence between an integrated source and destination market is not economically
plausible, so part of the Polish trend is likely a measurement artefact.

The inference has support. Eurostat's 2017 review of national approaches found "highly
divergent results across EU member states, largely driven by the different methodologies
used". Used cars are the textbook hard case — the vehicle ages between observations, and
whether ageing counts as a quality change determines the index's sign. The UK prices a fixed
sample of one-, two- and three-year-old cars and interpolates so that cars of the same age
and mileage are priced each month; Manheim likewise builds an index independent of shifts in
the characteristics of vehicles sold. Poland's national HICP metadata documents general
quality-adjustment methods but **says nothing specific about second-hand cars** — which
method GUS applies could not be determined, and that is a real gap.

Note also that CP07112 shows only +5.9 % across 2021→2023 while Indicata shows a ~+20–24 %
boom over the same span. The series that most under-detects the boom is the same one showing
the deepest subsequent fall.

## 3. Segment drift is large enough to break a single scalar

Indicata, Poland, retail price index Jan 2020 = 100, by fuel type (read from chart labels,
approximate):

- Petrol ≈ **110.6**, MHEV ≈ 110.4, diesel ≈ 108, HEV ≈ 105
- PHEV ≈ **92.1**
- **BEV ≈ 74.0**

A ~36 pp spread inside one country. European commentary is consistent: BEV values slip under
heavy new-car discounting while petrol holds and diesel proves "more resilient than expected
in Northern and Eastern Europe".

Drift is age-dependent too: in Poland, vehicles aged 31–90 months saw the largest drops
while the youngest and the oldest/cheapest stabilised; Europe-wide, 3–4-year-old cars are
scarce because of the 2020–2023 production gap.

So a single scalar is defensible for the ICE bulk of this dataset and **not** defensible for
the EVs the project deliberately keeps (`vol_engine == 0`), nor for the 3–7 year age band.
No evidence either way was found on **brand-tier drift** in Poland — an unfilled gap, not a
null result.

## 4. Failure modes of naive index rescaling

**a) Average transaction price is not a price index.** AAA AUTO's Poland median went
23 000 → 29 700 PLN from 2022 to 2023 (+29 %), but median vehicle age fell 13.5 → 12.7 years
and median mileage 188 000 → 181 000 km over the same period, which AAA AUTO attributes to an
influx of younger cars. Most of that +29 % is mix, not price. Every series built on mean or
median asking/transaction price (AAA AUTO, OTOMOTO Barometr, AutoCentrum) is mix-contaminated
and is the wrong instrument for rescaling a model that already conditions on age, mileage and
model. Constant quality is exactly what the RPPI Handbook and Manheim's methodology formalise.

**b) Double-counted depreciation — the bug this codebase is exposed to.** `REFERENCE_YEAR`
feeds `age = REFERENCE_YEAR - year`. Advancing it to 2026 takes a 2015 car from `age` 9 to
11, so the model *already* applies two more years of depreciation; an index that itself
embeds ageing then counts it twice. **`age` must own "the car got older"; the index must own
"the market moved", and must therefore be constant-age.**

**c) Multiplicative rescale interacts with `log1p`.** `expm1(ŷ) · k ≠ expm1(ŷ + ln k)`
because of the unit offset. Negligible at PLN scales, but it is a real ambiguity about
*where* the factor applies — pick one (post-`expm1` multiplication is the honest reading of
"convert to today's money") and document it.

**d) No per-row date means no true deflation.** The dataset has no observation date, so rows
cannot be deflated to a common base — only a single effective vintage can be assumed for the
whole training set. At ~10 %/yr drift, an unknown collection window is ±10 % on its own. This
is a hard ceiling on achievable accuracy and should be stated as one.

**e) Sources revise and get renamed.** Indicata warns its index is being revised. Eurostat
re-referenced all HICP indices to 2025 = 100 in February 2026, migrated to ECOICOP v2, and
flags `prc_hicp_midx` as discontinued in favour of `prc_hicp_minr` — confirmed live:
`unit=I15` still returns data, `unit=I25` returns an empty payload with a discontinued note.
A pinned fetch script will break silently.

**f) Degradation is not linear in elapsed time.** Vela et al. (Scientific Reports 12:11654,
2022) measured temporal degradation in 91 % of models across four industries and found the
patterns non-monotonic and dataset-specific. "The model is four years old, so widen by
4 × annual error" is not supported.

## 5. The uncertainty band

Strongest to weakest for this case:

- **Conformal prediction beyond exchangeability** (Barber, Candès, Ramdas, Tibshirani, *Ann.
  Statist.* 51(2) 2023) is the right formal tool: it drops exchangeability and reweights
  calibration residuals by recency. The earlier weighted-conformal-under-covariate-shift
  result (arXiv:1904.06019) assumes `Y|X` invariant — precisely what fails here, since the
  price relationship moved, not just the car mix. Cite the former, do not over-claim the latter.
- **Mass appraisal practice** (IAAO standards) requires adjustments to be "consistent and
  transparent" with documented derivation — while giving essentially no prescriptive
  methodology for time adjustments. Useful cover for a defensible-but-judgemental choice.
- **What to actually ship: publish the disagreement, not a point factor.** An HICP-derived
  factor of 0.76 against an Indicata-derived 0.90 for a mid-2022 vintage is a real, sourced
  band. An interval spanning two independent published indices is more honest, and more
  defensible, than a spuriously precise scalar.

## 6. Programmatic access

| Series | Publisher | Freq | Granularity | Access | Licence |
|---|---|---|---|---|---|
| HICP CP07112 (second-hand cars) | Eurostat / GUS | monthly | national | REST, verified: `ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_midx?format=JSON&geo=PL&coicop=CP07112&unit=I15` (JSON only; SDMX-CSV and multi-`geo` returned HTTP 400) | CC BY 4.0 / Decision 2011/833/EU |
| Same, mirrored | DBnomics | monthly | national | `api.db.nomics.world/v22/series/Eurostat/prc_hicp_midx/M.I15.CP07112.PL` — **more reliable than Eurostat's own API in testing**, stable IDs | mirror of source |
| HICP item weights | Eurostat | annual | national | `prc_hicp_inw`; PL second-hand cars 15.36‰ (2023), 12.10‰ (2024), 17.11‰ (2025) — note the volatility | same |
| Indicata Market Watch | Autorola | monthly | 16 countries, **by fuel type** | free PDF, no API | not stated → cite, do not redistribute |
| AAA AUTO Barometr | AAA AUTO | monthly | national + **all 16 voivodeships**, mean & median transaction price | press releases only | not stated |
| OTOMOTO Barometr | OTOMOTO | monthly | by province, median listing | news posts only | not stated |
| GUS BDL | GUS | varies | to locality | `bdl.stat.gov.pl/api/v1/`, JSON/XML, 5 req/s anon | **CC BY 4.0** |

The only series that is monthly, free, machine-readable, licensed for reuse **and** specific
to used cars is Eurostat CP07112 — which is also the one with the most doubt attached.
Everything with better market fidelity is PDF or commercial. That tension is the real
constraint.

GUS could not be confirmed to publish a used-car-specific index through BDL or DBW; Poland's
route to a published index appears to be its HICP submission to Eurostat.

## Decision taken in ADR 0001

The as-of date becomes a **first-class model output**, derived from the data's vintage. Any
index adjustment is a separate, clearly-labelled, toggleable layer publishing the
disagreement between two independent sources — never folded into the headline number.
Publishing "today's price = 2022 model × index" is *less* honest than publishing a clearly
dated 2022 valuation, because the multiplier carries an implicit claim of currency the model
cannot back.

## Gaps and uncertainties

- Why Polish CP07112 diverges from Germany and Czechia is **unresolved**. The
  "largely artefact" reading is inference, and should be treated as a reason not to rely on
  that series alone rather than as proof it is wrong.
- Indicata's 2021–2023 Poland values were read off a chart image; the Oct 2025 figure is
  from a printed table and is solid, the ~120–123 peak is approximate.
- **Brand-tier drift: no evidence found** — do not assume a single scalar works across tiers
  on the basis of absent evidence.
- **Province-level drift is unquantified**, which is a live question now that `province` is a
  real feature. The available OTOMOTO figures are mix-contaminated medians, not indices.
- The dataset's true vintage carries its own ±~10 % on any base period.
- **No published account was found** of a used-car ML model being recalibrated by index in
  practice; the closest documented analogues are real-estate mass-appraisal time adjustments
  and constant-quality index construction. The transfer is reasoning, not cited precedent.

## Sources

Eurostat dissemination API (`prc_hicp_midx`, `prc_hicp_manr`, `prc_hicp_inw`; PL/DE/CZ,
CP00 / CP0711 / CP07112) · DBnomics mirror · Eurostat HICP Methodological Manual 2024
(KS-GQ-24-003) and copyright policy · UK Statistics Authority APCP-T(22)03 *Second hand cars*
(citing the 2017 Eurostat review) · Indicata *Market Watch* Lite eds. 61 and 69 (Autorola) ·
Autovista24 / JD Power on the Polish used-car market · AAA AUTO *Barometr* · OTOMOTO
*Barometr* · AutoCentrum Q1 2026 segment analysis · GUS BDL and DBW API docs · GUS national
HICP metadata (`prc_hicp_esmshi4_pl`) · Manheim *Summary Methodology for the Used Vehicle
Value Index* · Eurostat/OECD/IMF *Handbook on Residential Property Price Indices* (2013),
ch. 5 · IAAO *Standard on Verification and Adjustment of Sales*, *Standard on Mass Appraisal
of Real Property* · Vela et al., *Temporal quality degradation in AI models*, Sci. Rep.
12:11654 (2022) · Tibshirani et al., *Conformal Prediction Under Covariate Shift*
(arXiv:1904.06019) · Barber et al., *Conformal prediction beyond exchangeability*, Ann.
Statist. 51(2) 2023
