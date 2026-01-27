*****************************************************
* Belt and Road Initiative (BRI) and trade volumes
* Difference-in-Differences with dynamic effects
* Data file: bri_trade_panel.dta
* Variables used:
*   province   - province code (31 provinces)
*   year       - 1978–2018
*   trade      - total imports + exports
*   primary    - GDP of primary industry
*   secondary  - GDP of secondary industry
*   tertiary   - GDP of tertiary industry
*   treat      - 1 if province affected by BRI, 0 otherwise
*****************************************************

clear all
set more off

*----------------------------------------------------
* 0. Set working directory and load data
*    (CHANGE the path to your local folder)
*----------------------------------------------------
cd "YOUR_PATH_TO_BRI_DATA"

use "bri_trade_panel.dta", clear

* Panel declaration
xtset province year

*****************************************************
* 1. Generate DID variables
*****************************************************

* Policy year: 2013
gen post = (year >= 2013)
label var post "1 after 2013 (post-BRI)"

gen did = treat * post
label var did "treat × post"

*****************************************************
* 2. Baseline DID regression
*****************************************************

* Two-way FE DID with clustered SE at province level
capture which reghdfe
if _rc ssc install reghdfe

reghdfe trade did primary secondary tertiary, ///
    absorb(province year) vce(cluster province)

* If reghdfe is not available, you can use:
* xtreg trade did primary secondary tertiary i.year, ///
*     fe vce(cluster province)

*****************************************************
* 3. Dynamic DID (parallel trends / event study)
*****************************************************

* Event time relative to 2013 for treated provinces
gen event_time = year - 2013 if treat == 1

* Leads: pre-treatment years (omit pre-1 as base)
forvalues k = -7/-1 {
    local name = "pre" + string(abs(`k'))
    gen `name' = (event_time == `k') & treat == 1
}

* Current treatment year
gen current = (year == 2013) & treat == 1

* Lags: post-treatment years
forvalues k = 1/5 {
    local name = "post" + string(`k')
    gen `name' = (event_time == `k') & treat == 1
}

* Drop pre-1 so that coefficients are relative to the year just before BRI
drop pre1

* Event-study regression
reghdfe trade pre2-pre7 current post1-post5 ///
    primary secondary tertiary, ///
    absorb(province year) vce(cluster province)

* (You can later extract coefficients to draw the parallel trends graph.)

*****************************************************
* 4. In-time placebo DID (fake policy year)
*****************************************************

* Example placebo: assume fake policy year 2008
gen post_plac = (year >= 2008)
gen did_plac  = treat * post_plac

reghdfe trade did_plac primary secondary tertiary, ///
    absorb(province year) vce(cluster province)

*****************************************************
* 5. In-space placebo DID (fake treated provinces)
*****************************************************

* Randomly assign fake treatment status, keeping share of treated similar
set seed 12345

preserve

tempvar treat_fake
gen `treat_fake' = treat

bysort year: gen u = runiform()
gsort year u
by year: gen treat_fake = treat[_n]

gen did_fake = treat_fake * post

reghdfe trade did_fake primary secondary tertiary, ///
    absorb(province year) vce(cluster province)

restore

*****************************************************
* End of file
*****************************************************
