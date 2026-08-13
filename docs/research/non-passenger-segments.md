# Extending Beyond Passenger Cars — Is There Any Data?

Date: 2026-08-12
Status: accepted
Author: P0w3r223
Related to: [ADR 0001](../adr/0001_scope-and-site-expansion.md) (conditional segment stream),
[price-data-sources.md](price-data-sources.md)

---

Research question: can a Polish used-vehicle price model be extended to light commercial
vehicles, motorcycles, trucks or agricultural machinery using data that may legally be
published in an open portfolio project — and would those segments need their own models
rather than a `segment` feature added to the existing one?

**Answer: a firm no for motorcycles, trucks and agriculture; a conditional maybe for vans.
And they would need separate models regardless.**

## 1. Poland has no per-unit price data for any non-car segment under an acceptable licence

This is a researched negative, not a shortfall of effort.

- The national open-data catalogue, queried for "pojazdy", returns registrations by type and
  voivodeship, technical-inspection results, driving-licence counts and the CEPiK API —
  **zero price, sale, auction or valuation datasets**.
- **CEPiK** is the best Polish vehicle data available: 60+ parameters per vehicle, free,
  reusable with attribution, covering *all* categories including motorcycles, trucks and
  agricultural tractors. It has **no price and no transaction value of any kind**. It is a
  covariate and market-structure source, never a target source. (The two official pages
  disagree on whether the licence is CC BY 4.0 or CC BY-SA 4.0 — worth resolving before use.)
- **Bailiff and tax-office auctions** do publish money figures for trucks and machinery, but
  as HTML notices with no bulk export or open-data licence, and the published figures are
  *suma oszacowania* / opening price — typically 50–70 % below market. Modelling that target
  produces a forced-sale appraisal model wearing a market-price label.
- Every Polish vehicle dataset on Kaggle is passenger cars. GitHub's otomoto-scraper topic is
  likewise car-category only — and is the route this project rejected.

## 2. Closest to feasible: light commercial vehicles, via one European dataset

The **AutoScout24 2025 snapshot** on Zenodo/Kaggle is the only recent European listing
dataset found whose schema explicitly carries `vehicle_type ∈ {car, van, truck}`: ~120 000
listings, ~80 fields (pricing, specs, fuel/energy, equipment, location, seller), collected
2025-11-08, published 2025-11-18, declared **MIT**. AutoScout24's market list includes
Poland, so Polish rows may exist — unverified, and Poland is Otomoto's home turf, so expect a
thin tail.

The caveat is the one already stated for the second-dataset stream: **an uploader's MIT grant
does not extinguish the marketplace's sui generis database right or its terms of service.**
MIT is a software licence applied to a scraped corpus, and is legally weaker than the CC0 on
the current dataset. The stream should stay conditional on that licence verification and
inherit its answer.

## 3. Non-Polish price data exists, and each option has a disqualifying defect

| Source | Segments | Price? | Licence | Defect |
|---|---|---|---|---|
| Craigslist Used Cars | pickup, truck, van, bus, offroad | asking price | CC0, 426 880 rows | US, ~2021; **"truck" = pickup, not HGV**; motorcycles absent |
| Copart / IAAI (Rebrowser) | cars, **motorcycles**, **commercial trucks**; 2.5M records, daily | **no sold price** — only current high bid + estimated retail | free for research, paid for commercial | salvage: the target is damage-conditioned, not retail |
| GSA Auctions API | federal surplus incl. heavy machinery | `HighBidAmount` | US public domain | **no vehicle attributes** — a 69-char item name and a link |
| Seattle fleet surplus | trucks, heavy equipment | none | public domain | 63 rows |
| Used bikes India | motorcycles | asking price, ~32k | unverified | Indian market; transfer to Poland implausible (brand mix, A2 regime, incomes) |
| Tractors (Purdue, AETR) | agricultural | auction results 2020–2022 | papers CC-BY, **data is TractorZoom proprietary** | no open tractor price dataset found |

Two mirror-image findings worth naming: the largest current auction dataset (Copart) has
attributes but **no sale price**, and the cleanest public-domain price source (GSA) has price
but **no attributes**. Neither works alone, and they cannot be joined.

## 4. These are different pricing problems, not a `segment` one-hot

- **The target itself changes.** Polish LCV and truck listings quote *netto* and *brutto*
  interchangeably. One `price` column would silently mix two targets ~23 % apart — a defect
  the log-target discipline cannot fix.
- **Feature sets barely overlap.** Trucks: Euro emission class (urban access bans make this a
  cliff, not a slope), axle configuration, cab, retarder. Tractors: engine *hours* and PTO
  horsepower — kilometres do not exist. Motorcycles: bike type and **A2 compliance** (≤35 kW,
  a category live in Poland since 2013) as a discrete demand premium. Vans: payload,
  wheelbase, roof height, body conversion.
- **Depreciation shape differs.** Tractors are sharply non-linear in usage: ≈ $118 lost per
  hour below 500 hours against ≈ $2 per hour above 10 000. Motorcycles lose 15–30 % in year
  one and ~50 % by years 3–5, with a documented spring/summer selling premium — a seasonality
  feature cars in this dataset do not need. Vans lose 20–25 % in year one and sit at 40–50 %
  of list by year five, the shape closest to cars, which is why LCV is the least disruptive
  extension.
- **Market geography changes.** Over 40 % of Western-European heavy commercial vehicles are
  exported eastward, with Poland a top destination. `province` — the feature this project is
  building around — is close to meaningless for HGVs, where the unit is country and export
  corridor.
- **Target encoding would actively break.** OOF encoding of `mark`/`model` is
  segment-specific: Mercedes-as-car and Mercedes Sprinter/Actros share a token but not a price
  distribution, and ~118k car rows would dominate the encoding maps against any plausible
  van tail. This is the strongest technical argument for separate models.

The modelling approach itself does transfer — gradient boosting is already the published
method for heavy-equipment residual value. Only the data does not.

## 5. What *is* cleanly available, if the goal is coverage rather than a second model

CEPiK gives per-vehicle attributes and registration volumes across every segment under a
CC-BY-family licence — e.g. 21 395 used tractors registered in Poland in 2025, −10.3 % year
on year. That supports segment-level market context, fleet-age distributions and a documented
"why we stopped here" section, at zero legal risk.

Note that index calibration does **not** generalise either: HICP has a second-hand *motor
car* index, but COICOP 07.1.2 covers *new* motorcycles and bicycles — there is no second-hand
motorcycle index to calibrate against. See
[temporal-recalibration.md](temporal-recalibration.md).

## Gaps and uncertainties

- **Not verified:** the country breakdown and van/truck row counts inside the AutoScout24
  snapshot. Everything about it as an LCV source hinges on that — it needs a download and a
  `groupby(country, vehicle_type)` before any planning decision.
- **Not verified:** licences of the Kaggle used-bike sets and the 208k Polish car set —
  Kaggle's licence field does not render to fetchers.
- **Could not confirm** whether the EU Combined Nomenclature splits new from used at
  subheading level for 8704 (goods vehicles), 8711 (motorcycles) and 8701 (tractors) as it
  does for 8703 (tariff sites returned 403). If that split exists, Eurostat Comext would
  yield clean per-segment average unit values of used imports — aggregate only, but legally
  spotless and a plausible calibration anchor. Worth one focused check.
- **No reliable evidence found** for any open per-unit price dataset covering European heavy
  trucks or agricultural machinery, in any licence. Ritchie Bros, Mascus, TractorZoom and
  Autoline hold this data commercially. For those segments the honest answer is that no usable
  source exists.
- **Motorcycles in Poland: no source.** The only motorcycle price data found under an open-ish
  licence is Indian, or US salvage without sold prices.

## Sources

dane.gov.pl datasets API (q=pojazdy) · CEPiK API and its gov.pl / Ministry of Digitisation
pages · Portal Obwieszczeń i Licytacji Komorniczych · Kaggle: `bartoszpieniak/poland-cars-for-sale-dataset`,
`austinreese/craigslist-carstrucks-data`, `rebrowser/copart-dataset`, `saisaathvik/used-bikes-prices-in-india` ·
GitHub topic `otomoto-scraper` · Zenodo record 17643343 (AutoScout24 2025 snapshot, MIT) and
AutoScout24 company market list · GSA Auctions API field reference · data.seattle.gov Current
Fleet Surplus/Auction List · Purdue *A hedonic analysis of farm tractor auction prices* (CC-BY,
2026) · AETR/AEEE *Hedonic Price Analysis of Used Tractors* (2023) · Farmer.pl on 2025 used
tractor registrations · AJOT on Europe's used-truck export landscape · AI Online on used-truck
price drivers · webBikeWorld on motorcycle depreciation · Loads of Vans on van depreciation ·
getruck.eu (netto quoting) · Rankomat on the A2 category · Fleet News on used LCV values (June
2025) · Statistics Poland HICP/COICOP 2018 methodology · ScienceDirect, ML for residual value
of heavy construction equipment (S0926580521002788)
