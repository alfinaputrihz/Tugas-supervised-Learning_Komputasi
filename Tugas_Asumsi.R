#persiapan data
baca_data <- fread("C:/Users/ACER/Downloads/Supervised Learning Komputasi/raw_data_simulasi.txt")
set.seed(12)
df <- baca_data[sample(nrow(baca_data), 5000), ]
head(df)

library(lmtest)
library(car)
library(nortest)
library(nlme)
library(MASS)
library(glmnet)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(knitr)
library(Metrics)

# ============================================================
# a. PENGUJIAN 4 ASUMSI DENGAN SEMUA 30 VARIABEL
# ============================================================
mod_ols <- lm(Y ~ ., data = df)  
res_ols <- residuals(mod_ols)

# 1. Normalitas Sisaan 
shapiro.test(res_ols)

# 2. Multikolinearitas 
vif_hasil <- vif(mod_ols)
print(vif_hasil)
any(vif_hasil > 10)  

# 3. Heteroskedastisitas
bptest(mod_ols)      

# 4. Autokorelasi Sisaan
dwtest(mod_ols)      


# ============================================================
# b. PENANGANAN DENGAN 4 MODEL
# ============================================================

# Model 2: WLS (Penanganan HETEROSKEDASTISITAS)
# Menghitung bobot invers varians
w <- 1 / (fitted(lm(abs(res_ols) ~ fitted(mod_ols)))^2)
mod_wls <- lm(Y ~ ., data = df, weights = w)

# Model 3: GLS (Penanganan AUTOKORELASI & HETEROSKEDASTISITAS)
# Menggunakan struktur autokorelasi AR(1)
mod_gls <- gls(Y ~ ., data = df, correlation = corAR1())

# Model 4: RIDGE REGRESSION (Penanganan MULTIKOLINEARITAS 30 VARIABEL)
X <- model.matrix(Y ~ ., data = df)[, -1]  

cv_ridge  <- cv.glmnet(X, y, alpha = 0)
best_lam  <- cv_ridge$lambda.min
mod_ridge <- glmnet(X, y, alpha = 0, lambda = best_lam)


# ============================================================
# c. PERBANDINGAN MODEL (RMSE & AIC)
# ============================================================
rmse <- function(act, pred) sqrt(mean((act - pred)^2))

# Prediksi tiap model
pred_ols   <- predict(mod_ols)
pred_wls   <- predict(mod_wls)
pred_gls   <- predict(mod_gls)
pred_ridge <- as.vector(predict(mod_ridge, newx = X))

# AIC untuk Ridge
aic_ridge <- function(act, pred, p) {
  n <- length(act); rss <- sum((act - pred)^2)
  n * log(rss / n) + 2 * (p + 1)
}

# Matriks Hasil Perbandingan
tabel_evaluasi <- data.frame(
  Model = c("OLS (Standar)", "WLS (Atasi Hetero)", "GLS (Atasi Autokor)", "Ridge (Atasi Multikol)"),
  RMSE  = c(rmse(y, pred_ols),
            rmse(y, pred_wls),
            rmse(y, pred_gls),
            rmse(y, pred_ridge)),
  AIC   = c(AIC(mod_ols),
            AIC(mod_wls),
            AIC(mod_gls),
            aic_ridge(y, pred_ridge, p = ncol(X)))
)

print(tabel_evaluasi)
