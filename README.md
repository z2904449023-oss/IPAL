# Coding Samples – Stata Empirical Work & News Sentiment Pipeline  
Author: Xinyi Zhou

This repository contains three Stata projects that illustrate my experience with applied econometrics and reproducible workflows during IPAL, and one Python project that builds a daily Crash/Fomo news sentiment index from Factiva/Dow Jones articles.  
The focus is on clear model specification, transparent coding, robustness checks, and end-to-end data pipelines.

Raw data files are **not** included due to licensing and confidentiality, but each project is fully runnable once the corresponding data files are placed in the indicated folders.

---

## 1. CFPS Smoking Behaviour and Heckman Selection (`cfps_smoking/`)

**File:** `cfps_smoking.do`  
**Data:** `Panel Data(CFPS10-16).dta`  
**Dataset:** CFPS 2010–2016 individual-level panel (China Family Panel Studies), restricted-access micro data.

**Goal**  

Study the determinants of smoking behaviour and smoking intensity, and illustrate how sample selection bias can be addressed using a Heckman selection model.

**Methods**

- Linear probability model (OLS) as a baseline for the binary outcome `smoke`.
- Logit and probit models for the smoking decision with:
  - Robust standard errors  
  - Marginal effects via `margins`
- Predicted probabilities from OLS/logit/probit for comparison.
- Heckman selection model for the intensive margin:
  - Outcome: number of cigarettes per day (`num_smoke`)  
  - Selection equation: smoking decision (`smoke`)  
  - Comparison of OLS on smokers only vs. Heckman

**How to run**

1. Place `Panel Data(CFPS10-16).dta` in the same directory as the `.do` file.  
2. Open `cfps_smoking.do` in Stata.  
3. Edit the `cd "YOUR_PATH_TO_CFPS_DATA"` line to your local path.  
4. Run the entire script.  
5. The script installs required user-written commands (e.g. `outreg2`, `asdoc`) if needed and produces regression tables and summary statistics.

---

## 2. Class Size and Achievement RDD (`class_size_rd/`)

**File:** `class_size_rd.do`  
**Data:** `lec4_grade.dta`  
**Dataset:** Angrist & Lavy–style school-level data on class size and test scores.

**Goal**

Estimate the causal effect of class size on student achievement using a regression discontinuity design at the class-size cutoff (40 students), and perform standard RD diagnostics and robustness checks.

**Methods**

- Baseline OLS regressions of average math scores (`avgmath`) on class size (`classize`) and controls (`disadv`, `enrollment`, `esquare`).  
- Manual local polynomial RD around the cutoff (enrollment between 35–45):
  - Left/right local regressions and calculation of the intercept jump.
- `rdrobust` local linear RD:
  - Triangular kernel, specified bandwidth (`h=5`, `h=6`)  
  - Covariate adjustment with `disadv`
- `rdplot` visualisation of the RD:
  - Different kernel choices (triangular vs. uniform)  
  - Different polynomial orders (`p=1`, `p=2`)
- Density test for manipulation of the running variable using `DCdensity`.  
- Placebo tests:
  - Placebo outcome (`avgverb`)  
  - Placebo cutoff (c = 30)

**How to run**

1. Place `lec4_grade.dta` in the same directory as the `.do` file.  
2. Open `class_size_rd.do` in Stata.  
3. Edit the `cd "YOUR_PATH_TO_GRADE_DATA"` line to your local path.  
4. Run the script. It will:
   - Install `rdrobust` and `DCdensity` if missing  
   - Produce RD estimates and RD plots  
   - Run placebo and robustness checks

---

## 3. Belt and Road Initiative (BRI) and Trade – DID (`bri_trade_did/`)

**File:** `bri_trade_did.do`  
**Data:** `bri_trade_panel.dta`  
**Dataset:** Province-level panel for China (31 provinces, 1978–2018), constructed from National Bureau of Statistics yearbooks (not included here).

**Goal**

Evaluate the impact of the Belt and Road Initiative (BRI) on provincial trade volumes using a difference-in-differences (DID) framework with fixed effects, event-study dynamics, and placebo tests.

**Variables used (in the data)**

- `province` – province identifier  
- `year` – year (1978–2018)  
- `trade` – total imports + exports  
- `primary` – primary industry value added  
- `secondary` – secondary industry value added  
- `tertiary` – tertiary industry value added  
- `treat` – indicator for BRI-affected provinces  

The `.do` file generates:

- `post` – post-BRI period indicator (year ≥ 2013)  
- `did` – DID treatment (`treat × post`)  
- Event-time dummies (`pre2`–`pre7`, `current`, `post1`–`post5`)

**Methods**

- Two-way fixed-effects DID with clustered standard errors:
  - `trade` on `did` and sectoral controls (`primary`, `secondary`, `tertiary`)  
  - Province and year fixed effects (`absorb(province year)`)
- Event-study (dynamic DID):
  - Leads and lags around the policy year (2013)  
  - Coefficients interpreted as trade changes relative to the year just before BRI  
  - Used to check parallel trend assumptions and dynamic policy effects
- In-time placebo:
  - Shift the policy year to a fake year (e.g. 2008)  
  - Re-estimate DID to test for spurious effects
- In-space placebo:
  - Randomly reassign treatment status across provinces within each year  
  - Re-estimate DID with `did_fake` to evaluate how unusual the true effect is under random assignment

**How to run**

1. Place `bri_trade_panel.dta` in the same directory as the `.do` file.  
2. Open `bri_trade_did.do` in Stata.  
3. Edit the `cd "YOUR_PATH_TO_BRI_DATA"` line to your local path.  
4. Run the script. It will:
   - Perform the baseline DID regression  
   - Estimate dynamic effects via event-study specification  
   - Run in-time and in-space placebo checks

---

## 4. News-based Crash & Fomo Sentiment Pipeline (`news_fomo_crash_pipeline/`)

**File:** `factiva_fomo_crash_pipeline.py`  
**Data:** Factiva / Dow Jones news exports in PDF, RTF, or CSV format

**Goal**

Build a daily time series of Crash and Fomo sentiment indices from financial news, using a combination of rule-based dictionaries and an LLM-based scorer, with caching and reproducible aggregation.  
This pipeline is used in my research on narrative-driven skewness risk premia and tail risk.

**High-level workflow**

1. **Input and extraction**
   - Accepts one of:
     - A batch directory (`--input_dir`) containing PDFs/RTFs/CSVs exported from Factiva/Dow Jones  
     - A single file (`--input_pdf`, `--input_rtf`, or `--input_csv`)
   - For PDFs:
     - Uses PyMuPDF to read pages and detect table-of-contents pages  
     - Parses TOC entries to infer article titles and page ranges  
     - Extracts article bodies and cleans boilerplate headers/footers
   - For RTF:
     - Converts RTF to plain text via heuristics  
     - Segments individual articles using title and word-count patterns
   - For existing CSVs:
     - Loads any file that already contains the required columns (`标题`, `起始页`, `结束页`, `正文`) and fills `source_file`, `source_path`, `source_type` if missing
   - All extracted articles are combined into a single DataFrame with unified columns.

2. **Metadata parsing and deduplication**
   - For each article, the script:
     - Normalizes the title and body text  
     - Extracts publisher, publication date, and time from the text (supporting both Chinese and English date formats)  
     - Computes a SHA-1 hash of `(title + body)` as a stable content identifier
   - Articles are deduplicated by `(publisher_norm, date, content_sha1)` to avoid counting the same article multiple times.

3. **Crash & Fomo scoring (Scheme C: Rule + LLM + fusion)**
   - **Rule-based scorer**
     - Uses curated Chinese/English phrase lists for Crash and Fomo, covering:
       - Direction (Crash vs no Crash, Fomo vs no Fomo)  
       - Intensity levels (0–3) for downside panic vs upside chase/frenzy  
       - Certainty levels (0–2) for how speculative vs realized the language is  
     - Detects patterns such as one-day percentage drops, selloffs, bear-market language, risk-on rotations, and buy-the-dip behaviour.
     - Converts direction, intensity, and certainty into normalized scores in [0, 1] for Crash and Fomo.
   - **LLM-based scorer**
     - Calls an OpenAI-compatible chat completion endpoint (default: DeepSeek-style API, configurable via `--base_url` or `OPENAI_BASE_URL`)  
     - Uses a detailed rubric (`LLM_SYSTEM_RUBRIC`) to instruct the model to return JSON labels:
       - `dir_crash`, `intensity_crash`, `certainty_crash`  
       - `dir_fomo`, `intensity_fomo`, `certainty_fomo`  
       - Short evidence quotes for each dimension
     - Enforces JSON output and post-processes the result to keep labels in valid ranges.
     - Caches results in a local SQLite database (`LLMCache`) keyed by `(model_id, content_sha1)` to avoid duplicate API calls.
   - **Fusion**
     - Combines rule and LLM scores using:
       - A rule weight parameter `--weight_rule` (default 0.3)  
       - An agreement tolerance `--agree_tol` for when to average vs. weighted-average
     - Handles cases where rule and LLM disagree in either:
       - Score levels (continuous Crash/Fomo scores)  
       - Directions (whether Crash/Fomo is detected at all)
     - Outputs fused scores and diagnostic flags:
       - `fused_crash_score`, `fused_fomo_score`  
       - `dir_conflict_crash`, `dir_conflict_fomo`  
       - `needs_review` indicator for potential manual audit

4. **Aggregation and outputs**
   - Aggregates per-article scores into daily indices:
     - `Crash_t` and `Fomo_t` based on:
       - Rule-only scores  
       - LLM-only scores  
       - Fused scores
   - Writes a single Excel workbook (`--output_xlsx`) with multiple sheets:
     - `daily_fused` – daily Crash/Fomo indices using fused scores  
     - `daily_rule` – daily indices using rule-only scores  
     - `daily_llm` – daily indices using LLM-only scores  
     - `articles_scored` – full per-article results with metadata, scores, and diagnostics
   - Optionally writes a combined extracted CSV (`--extract_csv`) with all intermediate article-level fields.

**How to run**

1. Install Python dependencies (example):

   ```bash
   pip install pandas tqdm pymupdf openpyxl openai

2. Set your API key and (if needed) base URL for an OpenAI-compatible endpoint:

  export OPENAI_API_KEY="YOUR_API_KEY"
  Optional if using a non-default base URL:
  export OPENAI_BASE_URL="https://api.deepseek.com"

3. Prepare your Factiva/Dow Jones exports (PDF/RTF/CSV) in a directory, e.g. data/factiva_raw/.

4. Run the pipeline. Example:
  python factiva_fomo_crash_pipeline.py \
    --input_dir data/factiva_raw \
    --output_xlsx results/news_fomo_crash_scores.xlsx \
    --extract_csv results/dowjones_news_combined.csv \
    --model deepseek-chat \
    --cache_db cache/llm_cache.sqlite \
    --weight_rule 0.3 \
    --agree_tol 0.2 \
    --workers 4

Useful options:

--disable_llm – run rule-only scoring (fused scores equal rule scores; no API calls).

--force_llm – ignore cache reads and always call the API (still writes cache).

--max_chars – limit the number of characters per article sent to the LLM (for latency/cost control).

--thinking – control DeepSeek-style “thinking” mode (disabled by default).

--no_response_format – fallback mode if response_format=json_object is not supported.
