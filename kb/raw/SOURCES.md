# Knowledge Base Sources — Rice Agronomy Corpus for Kanpur Advisory System

Assembly date: 2026-10-21.

## Primary sources

Each cited by section letter(s) in `kb_corpus.txt`:

| Source | URL | Used in sections |
|--------|-----|-----|
| IRRI Rice Knowledge Bank — Water Management module | http://www.knowledgebank.irri.org/step-by-step-production/growth/water-management | B1–B6 |
| Wikipedia — Rice | https://en.wikipedia.org/wiki/Rice | B7, C1, C4, C5, D1, D6, F1, F4, F5, G1, G2, I1 |
| Wikipedia — Rice production in India | https://en.wikipedia.org/wiki/Rice_production_in_India | B8, H1, H2, H4, H5, H6, I4 |
| Wikipedia — Rice blast | https://en.wikipedia.org/wiki/Rice_blast | D1, D2, D3 |
| Wikipedia — Brown planthopper | https://en.wikipedia.org/wiki/Brown_planthopper | C1, C2, C3 |
| Wikipedia — Sheath blight | https://en.wikipedia.org/wiki/Sheath_blight | D4, D5 |
| Wikipedia — Zinc deficiency (plant disorder) | https://en.wikipedia.org/wiki/Zinc_deficiency_(plant_disorder) | A4, J1 |
| Phytojournal 2018 (NDUAT Faizabad NPK field trial, Pusa Basmati 1) | https://phytojournal.com/archives/2018/vol7issue5/PartY/7-5-257-109.pdf | A1, A2, A3, A5, A6, H3, I2 |
| DRDPAT Rice Varieties database (page 04) | https://drdpat.bih.nic.in/Rice%20Varieties%20-%2004.htm | E4, E5 |
| Global Agriculture — List of notified rice varieties for UP and Uttaranchal | https://www.global-agriculture.com/seed-industry/list-of-notified-rice-varieties-in-uttar-pradesh-uttaranchal/ | E1, E2, E3 |
| NDUAT 2021 Varieties & Technologies PDF | https://mail.nduat.org/pdf/Varieties_Technologies_2021.pdf | E6 |
| IndiaAgronet — Rice (crop info) | https://www.indiagronet.com/indiaagronet/crop%20info/Rice.htm | B7, F2, H2 |

## Known limitations

1. **Hyperlocal Kanpur-specific content is thin.** CSAUAT Kanpur and KVK Kanpur publications were not available in English PDF form during corpus assembly. Content is scoped to "North Indian plains / eastern UP" with explicit Kanpur-applicability where soil pH, climate, and ecosystem match.

2. **Wikipedia is used as a consolidated reference** for well-established agronomic facts (growth stages, pest/disease biology, cultivation methods). Wikipedia is NOT cited in the paper's manuscript references — it functions here as a plain-English synthesis of primary agronomic literature. The paper cites primary sources (IRRI publications, FAO manuals, ICAR research) separately.

3. **Variety specifications** cover only Narendra-80 and Pant Dhan-6 in detail. Specifications for NDR 97, Sarjoo-52, and other named UP varieties are limited to their notification status; full agronomic specifications would require NDUAT/CSAUAT variety catalogues.

4. **Nutrient recommendations (120:60:60 NPK + 5 Zn)** are cited to one NDUAT Faizabad field trial. Broader UP state recommendations may vary ±10-15% around these values. The corpus is intended for advisory decision support, not prescriptive application.

5. **No supplementary pest ETL (Economic Threshold Levels)** are included because the Wikipedia articles consulted do not provide numeric ETLs. For precise ETL values, agronomist consultation is required.

## For future KB expansion

If/when the following become accessible, they should be added:
- CSAUAT (Chandrashekhar Azad University of Agriculture and Technology, Kanpur) annual package of practices
- KVK Kanpur Nagar and KVK Kanpur Dehat farmer advisory bulletins (English translations)
- ICAR-IIRR Hyderabad annual Package of Practices document
- UP Department of Agriculture seasonal advisory for Kharif rice
- FAO Rice Production Manual for South Asia (full PDF)
- Published Pest ETL compendium (e.g., Dhaliwal, Jindal and Dhawan 2010)

## Corpus-level provenance

- **Assembled by:** automated web retrieval + manual synthesis
- **Chunk count:** approximately 50 topical passages across 10 parts (A through J)
- **Approximate word count:** 5,400 words
- **Approximate size:** 37 KB
- **Target use:** RAG retrieval for farmer advisory LLM generation
- **Suggested chunk size for Phase 4:** 150-200 words with 30-word overlap, respecting the natural passage boundaries (## section headings) in this file.
