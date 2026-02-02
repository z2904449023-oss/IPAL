# Coding Samples – Stata Empirical Work & News Sentiment Pipeline  
Author: Xinyi Zhou

This project analyzes the impact of the Belt and Road Initiative (BRI) on provincial trade outcomes in China using panel data and modern difference-in-differences methods. The empirical strategy follows **Borusyak, Jaravel & Spiess (2021)** and includes both **temporal** and **spatial placebo tests** to assess robustness.

The analysis uses province-level annual panel data and implements:

* Data cleaning and missing-value checks
* Construction of treatment timing (`bri_year`)
* Event-time variable (`rel_year`)
* Descriptive trends for treated vs. untreated provinces
* DID-imputation estimation
* Temporal placebo (shift treatment 5 years earlier)
* Spatial placebo (randomly assigned treatment years)

---

## **Data**

Due to privacy policy, data is not public. Once you create your own data and add the path at the right place, the code could work.

### **Key variables**

| Variable           | Description                                    |
| ------------------ | ---------------------------------------------- |
| `prvn`             | Province identifier                            |
| `year`             | Calendar year                                  |
| `trade`            | Total import–export value                      |
| `export`, `import` | Components of trade                            |
| `bri_year`         | First BRI implementation year in each province |
| `rel_year`         | Event time relative to treatment               |
| `lntrade`          | Log of total trade                             |

Observations prior to 1988 or with missing trade variables are removed.

---

## **1. Data Cleaning & Panel Checks**

The script:

1. Summarizes variables and checks missing values
2. Drops observations missing `trade`, `export`, or `import`
3. Verifies panel structure using `xtset`
4. Detects potential gaps in year sequences
5. Restricts the dataset to 1988 onward

---

## **2. Constructing Treatment & Event-Time Variables**

The code assigns each province its BRI adoption year (`bri_year`) based on official rollout timing.

`rel_year = year – bri_year` is computed for event-study analysis.
Provinces never treated are coded as `bri_year = 0` and excluded from event-time calculations.

`lntrade = ln(trade)` is used as the outcome.

---

## **3. Descriptive Trends**

To visualize raw patterns independent of any model, the script:

* Groups provinces by BRI adoption year
* Collapses data to year-level means
* Generates a line plot comparing treated cohorts and never-treated units

This provides an intuitive check of pre-trend similarity.

---

## **4. DID-Imputation Estimation**

The main causal analysis applies:

**Borusyak, Jaravel & Spiess (2021) — “Imputation-based DID”**

Using:

```stata
did_imputation lntrade prvn year bri_year, ///
    pre(10) horizons(0/8) cluster(prvn) autos delta(1) minn(5)
```

This estimates dynamic treatment effects, using:

* 10 pre-treatment leads
* 0–8 post-treatment horizons
* Clustering at the province level

An event-study plot is generated with `event_plot`.

---

## **5. Temporal Placebo Test**

A falsification exercise shifts the treatment year **five years earlier**:

```
bri_year_T_Placebo = bri_year – 5
```

If significant effects appear **before** the true treatment date, it suggests potential violations of parallel trends.

Event-study estimates are plotted for comparison.

---

## **6. Spatial Placebo Test**

Another robustness test randomly assigns provinces into pseudo-treatment groups (2015–2018) using fixed random draws:

```stata
set seed 2024
gen rand = runiform() … (one value per province)
```

This intentionally destroys any true treatment variation.
If DID detects false "effects," this suggests model overfitting or spurious identification.

Results are again plotted using `event_plot`.

---

## **Outputs**

The script produces three event-study figures:

1. **Main DID-imputation result**
2. **Temporal placebo event-study**
3. **Spatial placebo event-study**

These visuals help assess whether treatment effects are credible and whether pre-trends and placebo tests support identification.

---

## **Reproducibility**

To reproduce all results:

1. Install required Stata packages

   ```stata
   ssc install did_imputation
   ssc install event_plot, replace
   ```
2. Ensure data file paths match your system
3. Run the do-file from top to bottom

---

## **Reference**

Borusyak, K., Jaravel, X., & Spiess, J. (2021). *Revisiting Event Study Designs: Robust and Efficient Estimation*. Working Paper.

---