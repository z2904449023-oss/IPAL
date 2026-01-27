*****************************************************
* Class size and achievement RD (Angrist & Lavy data)
* Data file: lec4_grade.dta
*****************************************************

clear all
set more off

*----------------------------------------------------
* 0. Set working directory and load data
*    (CHANGE the path to your local folder)
*----------------------------------------------------
cd "YOUR_PATH_TO_GRADE_DATA"

use "lec4_grade.dta", clear

*****************************************************
* 1. Baseline regressions
*****************************************************

reg avgmath classize, r
reg avgmath classize disadv, r

gen esquare = enrollment^2
reg avgmath classize disadv enrollment esquare, r

* Trim extreme enrollment values
drop if enrollment > 60
drop if enrollment < 20
reg avgmath classize disadv enrollment esquare, r

*****************************************************
* 2. Manual local linear RD around cutoff 40
*****************************************************

* Treatment indicator: small vs large class
gen largeclass = .
replace largeclass = 1 if enrollment <= 40
replace largeclass = 0 if enrollment > 40

* Left of cutoff: 35 ≤ enrollment < 40
reg avgmath largeclass disadv enrollment esquare if ///
    enrollment < 40 & enrollment >= 35, r
matrix coef_left = e(b)
local intercept_left = coef_left[1,5]

* Right of cutoff: 40 ≤ enrollment ≤ 45
reg avgmath largeclass disadv enrollment esquare if ///
    enrollment <= 45 & enrollment >= 40, r
matrix coef_right = e(b)
local intercept_right = coef_right[1,5]

local difference = `intercept_right' - `intercept_left'
macro list

*****************************************************
* 3. RD using rdrobust / rdplot and density test
*****************************************************

capture which rdrobust
if _rc ssc install rdrobust

* Main RD: triangular kernel, h = 5
rdrobust avgmath classize, c(40) p(1) q(2) ///
    covs(disadv) kernel(triangular) level(95) h(5) all

rdplot avgmath classize, c(40) p(1) kernel(triangular) ///
    graph_options(title(Figure) xtitle(enrollment) ///
                  ytitle(avgmath))

capture which DCdensity
if _rc ssc install DCdensity
DCdensity classize, breakpoint(40) generate(Xj Yj r0 fhat se_fhat)

*****************************************************
* 4. Placebo tests and robustness checks
*****************************************************

* Placebo outcome: verbal scores
rdrobust avgverb classize, c(40) p(1) q(2) ///
    covs(disadv) kernel(triangular) level(95) h(5) all

rdplot avgverb classize, c(40) p(1) kernel(triangular) ///
    graph_options(title(Figure) xtitle(enrollment) ///
                  ytitle(avgverb))

* Placebo cutoff: 30
rdrobust avgmath classize, c(30) p(1) q(2) ///
    covs(disadv) kernel(triangular) level(95) h(5) all

rdplot avgmath classize, c(30) p(1) kernel(triangular) ///
    graph_options(title(Figure) xtitle(enrollment) ///
                  ytitle(avgmath))

* Robustness: higher order polynomial
rdrobust avgmath classize, c(40) p(2) q(3) ///
    covs(disadv) kernel(triangular) level(95) h(5) all

rdplot avgmath classize, c(40) p(2) kernel(triangular) ///
    graph_options(title(Figure) xtitle(enrollment) ///
                  ytitle(avgmath))

* Robustness: change kernel to uniform
rdrobust avgmath classize, c(40) p(1) q(2) ///
    covs(disadv) kernel(uniform) level(95) h(5) all

rdplot avgmath classize, c(40) p(1) kernel(uniform) ///
    graph_options(title(Figure) xtitle(enrollment) ///
                  ytitle(avgmath))

* Robustness: change bandwidth to h = 6
rdrobust avgmath classize, c(40) p(1) q(2) ///
    covs(disadv) kernel(triangular) level(95) h(6) all

rdplot avgmath classize, c(40) p(1) h(6) kernel(triangular) ///
    graph_options(title(Figure) xtitle(enrollment) ///
                  ytitle(avgmath))
