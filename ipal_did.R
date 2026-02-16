## ====================== 0. Load packages ======================
library(haven)         # read Stata data (.dta)
library(dplyr)         # data manipulation
library(ggplot2)       # plotting
library(didimputation) # DID-imputation (Borusyak, Jaravel, Spiess)

## ====================== 1. Read data & basic checks ======================
# Equivalent to: use "...dta", clear
df <- read_dta("data.dta") # TODO: replace with your actual data path

# Simple summary statistics
summary(df)

# Check missing values for key variables
sapply(df[, c("trade", "export", "import")], function(x) sum(is.na(x)))

# Equivalent to: drop if missing(trade, export, import)
df <- df %>% filter(!is.na(trade), !is.na(export), !is.na(import))
summary(df)

# Check panel structure and gaps in year coverage within provinces
df <- df %>%
  arrange(prvn, year) %>%
  group_by(prvn) %>%
  mutate(gap = year - dplyr::lag(year)) %>%
  ungroup()

df %>%
  filter(gap > 1) %>%
  select(prvn, year, gap) %>%
  print(n = Inf)

# Equivalent to: drop if year < 1988
df <- df %>% filter(year >= 1988)
df <- df %>% select(-gap)

## ====================== 2. Construct bri_year, rel_year, lntrade ======================
# Equivalent to: gen bri_year = .; replace ... in Stata
df <- df %>%
  mutate(
    bri_year = NA_integer_,
    bri_year = ifelse(
      prvn %in% c(12, 13, 15, 22, 23, 32, 35, 36, 41, 42, 43, 44, 53, 62, 63, 64),
      2015L, bri_year
    ),
    bri_year = ifelse(prvn %in% c(34, 37, 51, 65), 2016L, bri_year),
    bri_year = ifelse(prvn %in% c(31, 45, 46, 50, 54), 2017L, bri_year),
    bri_year = ifelse(prvn %in% c(11, 21, 33, 52, 61), 2018L, bri_year),
    bri_year = ifelse(is.na(bri_year), 0L, bri_year) # never treated
  )

# Event time relative to BRI adoption
df <- df %>%
  mutate(
    rel_year = year - bri_year,
    rel_year = ifelse(bri_year == 0, NA, rel_year) # set to NA for never treated
  )

# Equivalent to: gen lntrade = ln(trade)
df <- df %>% mutate(lntrade = log(trade))

## ====================== 3. Descriptive trends (group means) ======================
# Define treatment group variable based on bri_year
# (kept here so that later placebo manipulations do not overwrite it)
df <- df %>%
  mutate(
    treat = case_when(
      bri_year == 2015 ~ 3L,
      bri_year == 2016 ~ 2L,
      bri_year == 2017 ~ 1L,
      bri_year == 2018 ~ 0L,
      bri_year == 0    ~ 4L,
      TRUE ~ NA_integer_
    )
  )

# Collapse to group-year averages
df_group <- df %>%
  group_by(treat, year) %>%
  summarise(mean_lntrade = mean(lntrade, na.rm = TRUE), .groups = "drop")

# Plot group-specific trends (similar to Stata twoway line ...)
ggplot(
  df_group,
  aes(
    x = year,
    y = mean_lntrade,
    group = factor(treat),
    linetype = factor(treat),
    color = factor(treat)
  )
) +
  geom_line() +
  scale_linetype_manual(
    values = c("0" = "solid", "1" = "solid", "2" = "solid", "3" = "solid", "4" = "dashed"),
    labels = c(
      "3" = "BRI provinces 2015 (treated)",
      "2" = "BRI provinces 2016 (treated)",
      "1" = "BRI provinces 2017 (treated)",
      "0" = "BRI provinces 2018 (treated)",
      "4" = "Never treated"
    )
  ) +
  scale_color_discrete(
    labels = c(
      "3" = "BRI provinces 2015 (treated)",
      "2" = "BRI provinces 2016 (treated)",
      "1" = "BRI provinces 2017 (treated)",
      "0" = "BRI provinces 2018 (treated)",
      "4" = "Never treated"
    )
  ) +
  labs(
    x = "Year",
    y = "Mean ln(trade)",
    title = "Mean trade: treated vs control",
    linetype = "",
    color = ""
  ) +
  theme_minimal()

## ====================== 4. DID-imputation (actual BRI timing) ======================
# In Stata: set bri_year to missing for never-treated;
# here we create g_bri_year for did_imputation
df <- df %>%
  mutate(
    g_bri_year = bri_year,
    g_bri_year = ifelse(bri_year == 0, NA, g_bri_year) # never treated = NA
  )

# Equivalent Stata call:
# did_imputation lntrade prvn year bri_year, pre(10) horizons(0/8) cluster(prvn) autos delta(1) minn(5)
# R version: pretrends = -10:-1, horizon = 0:8, cluster_var = "prvn"
did_real <- did_imputation(
  data = df,
  yname = "lntrade",
  gname = "g_bri_year",
  tname = "year",
  idname = "prvn",
  horizon = 0:8,
  pretrends = -10:-1,
  cluster_var = "prvn"
)
print(did_real)

# Event-study style plot using ggplot (instead of Stata event_plot)
did_real_es <- did_real %>%
  mutate(term = as.numeric(term))

ggplot(did_real_es, aes(x = term, y = estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point() +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  labs(
    x = "Years from treatment",
    y = "lntrade",
    title = "Borusyak et al. (2021) method of DID"
  ) +
  scale_x_continuous(breaks = -10:8) +
  theme_minimal()

## ====================== 5. Temporal placebo: shift all BRI years 5 years earlier ======================
# Equivalent to: gen bri_year_T_Placebo = bri_year; replace = bri_year-5 if bri_year>0; =. if bri_year==.
df <- df %>%
  mutate(
    g_bri_year_T = g_bri_year,
    g_bri_year_T = ifelse(
      !is.na(g_bri_year_T) & g_bri_year_T > 0,
      g_bri_year_T - 5L, g_bri_year_T
    )
  )

did_temporal <- did_imputation(
  data = df,
  yname = "lntrade",
  gname = "g_bri_year_T",
  tname = "year",
  idname = "prvn",
  horizon = 0:4,
  pretrends = -10:-1,
  cluster_var = "prvn"
)
print(did_temporal)

did_temporal_es <- did_temporal %>%
  mutate(term = as.numeric(term))

ggplot(did_temporal_es, aes(x = term, y = estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point() +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  labs(
    x = "Years to temporal placebo treatment",
    y = "lntrade",
    title = "Temporal Placebo: pseudo BRI 5 years earlier"
  ) +
  scale_x_continuous(breaks = -10:4) +
  theme_minimal()

## ===== 5(b). 100 temporal placebo random shifts + p-value + histogram =====
# First obtain the "true" tau0 (term == 0) under actual g_bri_year
did_real_tau0 <- did_imputation(
  data = df,
  yname = "lntrade",
  gname = "g_bri_year",
  tname = "year",
  idname = "prvn",
  horizon = 0,        # only tau0
  pretrends = NULL,
  cluster_var = "prvn"
)
theta_real <- did_real_tau0$estimate[did_real_tau0$term == "0"]

R <- 100
set.seed(2024)
theta_TP <- numeric(R)

years_range <- range(df$year, na.rm = TRUE)
ymin <- years_range[1]
ymax <- years_range[2]

for (r in 1:R) {
  # Random shift from -1 to -5
  k <- floor(runif(1) * 5) + 1
  shift <- -k

  df_sim <- df %>%
    mutate(
      g_bri_year_T100 = g_bri_year,
      g_bri_year_T100 = ifelse(
        !is.na(g_bri_year_T100) & g_bri_year_T100 > 0,
        g_bri_year_T100 + shift,
        g_bri_year_T100
      ),
      g_bri_year_T100 = ifelse(
        !is.na(g_bri_year_T100) & g_bri_year_T100 < ymin,
        ymin, g_bri_year_T100
      ),
      g_bri_year_T100 = ifelse(
        !is.na(g_bri_year_T100) & g_bri_year_T100 > ymax,
        ymax, g_bri_year_T100
      )
    )

  did_tp <- did_imputation(
    data = df_sim,
    yname = "lntrade",
    gname = "g_bri_year_T100",
    tname = "year",
    idname = "prvn",
    horizon = 0,
    pretrends = NULL,
    cluster_var = "prvn"
  )

  theta_TP[r] <- did_tp$estimate[did_tp$term == "0"]
}

# Randomization p-value (two-sided)
p_temporal <- (sum(abs(theta_TP) >= abs(theta_real)) + 1) / (R + 1)
cat("Temporal placebo RI p-value (two-sided) = ", p_temporal, "\n")

# Histogram of tau0 from temporal placebo simulations
df_theta_TP <- data.frame(theta_temporal = theta_TP)

ggplot(df_theta_TP, aes(x = theta_temporal)) +
  geom_histogram(
    aes(y = after_stat(100 * ..count../sum(..count..))),
    bins = 20, color = "black", fill = "grey80"
  ) +
  geom_vline(xintercept = theta_real, color = "red", linewidth = 1) +
  labs(
    title = "Temporal placebo (random shifts): distribution of tau0",
    x = "Estimated tau0 under random treatment years",
    y = "Percent"
  ) +
  theme_minimal()

## ====================== 6. Spatial placebo (random treated provinces) ======================
# Single spatial placebo run
set.seed(2024)
prov_rand <- df %>%
  distinct(prvn) %>%
  mutate(rand = runif(n()))

df_spatial <- df %>%
  left_join(prov_rand, by = "prvn") %>%
  mutate(
    g_bri_year_S = case_when(
      rand < 0.3 ~ 2015L,
      rand >= 0.3 & rand < 0.5 ~ 2016L,
      rand >= 0.5 & rand < 0.7 ~ 2017L,
      rand >= 0.7 & rand < 0.9 ~ 2018L,
      rand >= 0.9 ~ NA_integer_,
      TRUE ~ NA_integer_
    )
  )

table(df_spatial$g_bri_year_S, useNA = "ifany")

did_spatial <- did_imputation(
  data = df_spatial,
  yname = "lntrade",
  gname = "g_bri_year_S",
  tname = "year",
  idname = "prvn",
  horizon = 0:8,
  pretrends = -10:-1,
  cluster_var = "prvn"
)
print(did_spatial)

did_spatial_es <- did_spatial %>%
  mutate(term = as.numeric(term))

ggplot(did_spatial_es, aes(x = term, y = estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point() +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  labs(
    x = "Years to spatial placebo treatment",
    y = "lntrade",
    title = "Spatial Placebo: Randomly Assigned Provincial Treatment"
  ) +
  scale_x_continuous(breaks = -10:8) +
  theme_minimal()

## ===== 6(b). 100 spatial placebos + p-value + histogram =====
# Obtain the "true" tau0 again under actual g_bri_year
did_real_tau0_S <- did_imputation(
  data = df,
  yname = "lntrade",
  gname = "g_bri_year",
  tname = "year",
  idname = "prvn",
  horizon = 0,
  pretrends = NULL,
  cluster_var = "prvn"
)
theta_real_S <- did_real_tau0_S$estimate[did_real_tau0_S$term == "0"]

R_S <- 100
set.seed(2024)
theta_SP <- numeric(R_S)

for (r in 1:R_S) {
  prov_randS <- df %>%
    distinct(prvn) %>%
    mutate(randS = runif(n()))

  df_spatial_sim <- df %>%
    left_join(prov_randS, by = "prvn") %>%
    mutate(
      g_bri_year_S100 = case_when(
        randS < 0.3 ~ 2015L,
        randS >= 0.3 & randS < 0.5 ~ 2016L,
        randS >= 0.5 & randS < 0.7 ~ 2017L,
        randS >= 0.7 & randS < 0.9 ~ 2018L,
        randS >= 0.9 ~ NA_integer_,
        TRUE ~ NA_integer_
      )
    )

  did_sp <- did_imputation(
    data = df_spatial_sim,
    yname = "lntrade",
    gname = "g_bri_year_S100",
    tname = "year",
    idname = "prvn",
    horizon = 0,
    pretrends = NULL,
    cluster_var = "prvn"
  )

  theta_SP[r] <- did_sp$estimate[did_sp$term == "0"]
}

# Randomization p-value (two-sided) for spatial placebo
p_spatial <- (sum(abs(theta_SP) >= abs(theta_real_S)) + 1) / (R_S + 1)
cat("Spatial placebo RI p-value (two-sided) = ", p_spatial, "\n")

# Histogram of tau0 from spatial placebo simulations
df_theta_SP <- data.frame(theta_spatial = theta_SP)

ggplot(df_theta_SP, aes(x = theta_spatial)) +
  geom_histogram(
    aes(y = after_stat(100 * ..count../sum(..count..))),
    bins = 20, color = "black", fill = "grey80"
  ) +
  geom_vline(xintercept = theta_real_S, color = "red", linewidth = 1) +
  labs(
    title = "Spatial placebo (random provinces): distribution of tau0",
    x = "Estimated tau0 under random provincial assignment",
    y = "Percent"
  ) +
  theme_minimal()
