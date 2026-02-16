# Coding Samples – IPAL-DID 
Author: Xinyi Zhou

## **Overview**

This project evaluates the causal impact of the Belt and Road Initiative (BRI) on provincial trade flows in China using annual panel data and modern difference-in-differences methods. The analysis follows **Borusyak, Jaravel & Spiess (2021)** and supplements the main DID-imputation estimates with:

* **Temporal placebo tests**
* **Spatial placebo tests**
* **Randomization inference with 100 simulations** for both placebo types
* **Distributional inference for tau₀** (the contemporaneous treatment effect)

The code is written in Stata and R and is fully reproducible.

---

## **Stata**

## **Dataset**

The dataset is stored at:

```
/Users/zhouxinyi/Desktop/PKU/term paper/Data for Student Term Paper/Panel Data(China Provincial)2.dta
```

### **Key variables**

| Variable                    | Description                                   |
| --------------------------- | --------------------------------------------- |
| `prvn`                      | Province ID                                   |
| `year`                      | Observation year                              |
| `trade`, `export`, `import` | Trade variables                               |
| `bri_year`                  | The first year a province implemented the BRI |
| `rel_year`                  | Event time: `year – bri_year`                 |
| `lntrade`                   | Log of total trade                            |

Observations with missing trade data and those prior to 1988 are removed.

---

## **Code Structure**

### **1. Data inspection & missing-value handling**

The script:

* Summarizes variables
* Checks for missing data
* Removes observations missing `trade`, `export`, or `import`
* Verifies panel structure (`xtset prvn year`)
* Checks for gaps in yearly sequences
* Restricts dataset to years ≥ 1988

This ensures clean and balanced panel data for DID estimation.

---

### **2. Constructing treatment timing and event time**

The script assigns `bri_year` based on the adoption year of the BRI in each province.
Event time is defined as:

```
rel_year = year – bri_year
```

`lntrade = ln(trade)` is used as the main outcome.

---

### **3. Descriptive trends (pre-model visualization)**

To visually compare treated vs. untreated provinces:

* Provinces are grouped by BRI implementation year
* Yearly mean trade values are collapsed
* A multi-cohort trend plot is generated

This provides an intuitive check on trend similarity prior to treatment.

---

### **4. Main estimation: DID-Imputation (Borusyak et al., 2021)**

The core method uses:

```stata
did_imputation lntrade prvn year bri_year, pre(10) horizons(0/8) ///
    cluster(prvn) autos delta(1) minn(5)
```

Features:

* 10 pre-treatment leads
* 0–8 post-treatment horizons
* Province-clustered standard errors
* Automatic selection of untreated comparisons

An event-study plot is generated using `event_plot`.

---

### **5. Temporal Placebo Tests**

#### **(a) Single temporal placebo**

Treatment year is shifted **5 years earlier**:

```
bri_year_T_Placebo = bri_year – 5
```

This tests whether spurious effects appear **before** true policy implementation.

---

#### **(b) 100 temporal placebo simulations (random treatment shifts)**

To implement **randomization inference (RI)**:

* Random shifts of -1 to -5 years are applied to treated provinces
* `did_imputation` is run 100 times
* Each simulation extracts the placebo estimate of `tau0`
* These placebo τ₀ values form a **null distribution**

The script then:

* Computes a **two-sided RI p-value**:

  ```
  p = (# of |tau_placebo| ≥ |tau_real|) / (R + 1)
  ```
* Plots a histogram of simulated τ₀ values
* Marks the **real τ₀ estimate** with a red line

This is a rigorous falsification test.

---

### **6. Spatial Placebo Tests**

#### **(a) Single spatial placebo**

Provinces are randomly assigned into artificial treatment cohorts:

* 2015 group
* 2016 group
* 2017 group
* 2018 group

The distribution of random assignments approximates real treatment proportions.

---

#### **(b) 100 spatial placebo simulations**

Each simulation:

* Randomly assigns provinces to pseudo-treatment groups
* Runs `did_imputation`
* Extracts τ₀

A second RI p-value is computed, and a histogram is plotted.

This tests whether results could be driven by **geographical clustering or compositional differences**, rather than the BRI itself.

---

## **Outputs**

The script produces the following:

### **Event-Study Figures**

1. **Main DID-imputation event-study**
2. **Temporal placebo event-study**
3. **Spatial placebo event-study**

### **Randomization Inference Figures**

4. **Histogram of 100 temporal placebo τ₀ values**
5. **Histogram of 100 spatial placebo τ₀ values**

Both figures include a red reference line representing the **true** τ₀ estimate.

---

## **Reproducibility**

To reproduce the results:

1. Install required Stata packages:

```stata
ssc install did_imputation
ssc install event_plot, replace
```

2. Update the dataset path if needed
3. Run the do-file from top to bottom

The code uses fixed random seeds (`set seed 2024`), ensuring full replicability.

---

## **R**

---
Purpose:
   This script implements DID-imputation (Borusyak, Jaravel, 
   Spiess, 2021) for the impact of the Belt and Road Initiative
   (BRI) on provincial trade in China, using panel data
   and placebo tests (temporal and spatial).

 Data requirements:
   The input Stata file (data.dta) must contain at least:
     - prvn:   province identifier (numeric or integer)
     - year:   calendar year
     - trade:  trade volume (outcome)
     - export: export volume (used for missing-value filtering)
     - import: import volume (used for missing-value filtering)

 Main steps:
   0. Load required packages
   1. Read data and perform basic checks
   2. Construct treatment timing (bri_year), event time (rel_year),
      and log outcome (lntrade)
   3. Plot descriptive trends (group-year averages)
   4. Run DID-imputation with actual BRI timing
   5. Temporal placebo tests (single shift and random shifts)
   6. Spatial placebo tests (single random assignment and
      multiple random assignments)

 How to run:
   1. Install required packages (once):
        install.packages(c("haven", "dplyr", "ggplot2", "didimputation"))
   2. Replace the placeholder "data.dta" path below with the
      actual path to your Stata panel data file.
   3. Run the script in R / RStudio / VS Code.
---

## **References**

Borusyak, K., Jaravel, X., & Spiess, J. (2021).
*Revisiting Event Study Designs: Robust and Efficient Estimation.* Working Paper.

---
