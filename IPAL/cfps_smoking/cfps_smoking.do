*****************************************************
* CFPS 2010–2016: Smoking decision and intensity
* Data file: Panel Data(CFPS10-16).dta
*****************************************************

clear all
set more off

*----------------------------------------------------
* 0. Set working directory and load data
*    (CHANGE the path to your local folder)
*----------------------------------------------------
cd "YOUR_PATH_TO_CFPS_DATA"

use "Panel Data(CFPS10-16).dta", clear

* Install required user-written commands (if needed)
capture which outreg2
if _rc ssc install outreg2

capture which asdoc
if _rc ssc install asdoc

* Core covariates for smoking models
global xlist male age edu1 expense

*****************************************************
* Question 1: Binary choice model – smoking decision
*****************************************************

* Baseline OLS (for comparison)
reg smoke $xlist
est store reg

reg smoke $xlist, robust
est store reg_r

* Logit models with marginal effects
logit smoke $xlist
est store logit

margins, dydx(*)
est store margins_logit

logit smoke $xlist, robust
est store logit_r

margins, dydx(*)
est store margins_logit_r

* Probit models with marginal effects
probit smoke $xlist
est store probit

margins, dydx(*)
est store margins_probit

probit smoke $xlist, robust
est store probit_r

margins, dydx(*)
est store margins_probit_r

* Compare OLS, logit, probit and their marginal effects
est table reg reg_r logit margins_logit logit_r margins_logit_r ///
          probit margins_probit probit_r margins_probit_r, ///
          stats(N ll) b(%7.3f) stfmt(%8.2f)

* Predicted probabilities from different models
quietly logit smoke $xlist, robust
predict plogit, pr

quietly probit smoke $xlist, robust
predict pprobit, pr

quietly regress smoke $xlist, robust
predict pols, xb

asdoc summarize smoke plogit pprobit pols, replace

*****************************************************
* Question 2: Smoking intensity – Heckman selection
*****************************************************

* Descriptive statistics for number of cigarettes per day
asdoc summarize num_smoke, detail replace

* Keep only smokers and save a temporary file
use "Panel Data(CFPS10-16).dta", clear
keep if smoke > 0
save temp_data, replace

use temp_data, clear

* OLS on smokers only
reg num_smoke smoke male age edu1 expense
est store reg2

* Heckman selection model:
*   Outcome: num_smoke
*   Selection: smoke
heckman num_smoke male age edu1 expense, ///
    select(smoke = male age edu1 expense) nolog
est store heckman

* Compare OLS vs. Heckman
est table reg2 heckman, stats(N ll) b(%7.3f) stfmt(%8.2f)

outreg2 [reg2 heckman] using "comparison2.xls", ///
    stat(coef se) bdec(4) sdec(3) replace label
