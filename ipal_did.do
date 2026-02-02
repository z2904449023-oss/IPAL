* Data path example: use "/Users/zhouxinyi/Desktop/PKU/term paper/Data for Student Term Paper/Panel Data(China Provincial)2.dta",clear
*======================================================
* 1. Inspect basic dataset characteristics and Check missing values
*======================================================
sum
misstable summarize
drop if missing(trade, export, import)
sum

* Check panel structure and detect gaps in year coverage
xtset prvn year
bysort prvn (year): gen gap = year - year[_n-1]
list prvn year gap if gap > 1
drop if year < 1988
drop gap
*======================================================
* 2. Construct treatment timing (bri_year), event time (rel_year), and outcome (lntrade)
*======================================================
* bri_year denotes the first year in which each province implemented the BRI.
gen bri_year = . 
replace bri_year = 2015 if inlist(prvn, 12,13,15,22,23,32,35,36,41) ///
    | inlist(prvn, 42,43,44,53,62,63,64) 
replace bri_year = 2016 if inlist(prvn, 34,37,51,65)
replace bri_year = 2017 if inlist(prvn, 31,45,46,50,54)
replace bri_year = 2018 if inlist(prvn, 11,21,33,52,61)
replace bri_year = 0 if missing(bri_year)

* Event time relative to treatment
gen rel_year=year-bri_year
replace rel_year = . if bri_year == 0

* Log outcome
gen lntrade=ln(trade)

*======================================================
* 3. Descriptive trends of treated and untreated provinces
*======================================================

xtset prvn year
* Group provinces by their BRI adoption year
gen treat=bri_year
replace treat=3 if bri_year ==2015
replace treat=2 if bri_year ==2016
replace treat=1 if bri_year ==2017
replace treat=0 if bri_year ==2018
replace treat=4 if bri_year ==0


* Collapse to group-year averages for visualization
preserve
collapse (mean) mean_lntrade = lntrade, by(treat year)

twoway ///
    (line mean_lntrade year if treat==3, lpattern(solid)) ///
    (line mean_lntrade year if treat==2, lpattern(solid)) ///
	(line mean_lntrade year if treat==1, lpattern(solid)) ///
	(line mean_lntrade year if treat==0, lpattern(solid)) /// 
	(line mean_lntrade year if treat==4, lpattern(dash)), ///
    legend(order(1 "BRI provinces2015 (treated)" 2 "BRI provinces2016 (treated)" 3 "BRI provinces2017 (treated)" 4 "BRI provinces2018 (treated)" 5 "Never treated")) ///
    xtitle("Year") ///
    ytitle("Mean ln(trade)") ///
    title("Mean trade: treated vs control")
restore

*======================================================
* 4. DID-imputation & placebo test
*======================================================
ssc install did_imputation
ssc install event_plot,replace

replace bri_year = . if rel_year == .

preserve
did_imputation lntrade prvn year bri_year,pre(10) horizons(0/8) cluster(prvn) autos delta(1)minn(5)
est store did_real

event_plot, ciplottype(rcap) ///
    graph_opt( ///
        ytitle("lntrade") ///
        xtitle("Years from treatment") ///
        title("Borusyak et al. (2021) method of DID") ///
        xlabel(-10(1)8) ///
    )
restore

*Temporal placebo
gen bri_year_T_Placebo = bri_year
replace bri_year_T_Placebo = bri_year - 5 if bri_year >0 
replace bri_year_T_Placebo = . if bri_year == .

preserve  
did_imputation lntrade prvn year bri_year_T_Placebo,pre(10) horizons(0/8) cluster(prvn) autos delta(1) minn(5)
est store did_temporal

event_plot, ciplottype(rcap) ///
    graph_opt( ///
        yt("lntrade") ///
        xt("year from Temporal placebo treatment") ///
        t("Temporal Placebo: pseudo BRI 5 years earlier") ///
        xti("Years to Temporal placebo treatment") ///
        xla(-10(1)8) ///
    )
restore

*Spatial placebo
gen bri_year_S_Placebo = .
set seed 2024
egen tag = tag(prvn)
gen rand = runiform() if tag==1
bysort prvn: replace rand = rand[1]
replace bri_year_S_Placebo = 2015 if rand < 0.3
replace bri_year_S_Placebo = 2016 if rand >= 0.3 & rand < 0.5
replace bri_year_S_Placebo = 2017 if rand >= 0.5 & rand < 0.7
replace bri_year_S_Placebo = 2018 if rand >= 0.7 & rand < 0.9
tab bri_year_S_Placebo, nolabel missing

preserve
did_imputation lntrade prvn year bri_year_S_Placebo,pre(10) horizons(0/8) cluster(prvn) autos delta(1)minn(5)
est store did_spatial

event_plot, ciplottype(rcap) stub_lead(pre#) stub_lag(tau#)///
    graph_opt( ///
        ytitle("lntrade") ///
        xtitle("year from Spatial placebo treatment") ///
        title("Spatial Placebo: Randomly Assigned Provincial Treatment") ///
		xti("Years to Spatial placebo treatment") ///
        xlabel(-10(1)8) ///
    )
