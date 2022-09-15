**Table of Contents**

- [Package](#Package)
- [Installation](#Installation)
- [Usage](#Usage)
  * [LatentClassJM](#LatentClassJM)
  * [create.data](#createdata)
- [Help](#Help)
- [Contact](#Contact)
- [Reference](#Reference)


# Package
<ins>**LatentClassJM**</ins> is a package that performs the EM algorithm to compute the sieve non-parametric maximum likelihood estimator (NPMLE) for a semiparametric latent-class joint model, using the method proposed by [Wong et al.](https://doi.org/10.1214/21-AOS2117) (2022). 
**LatentClassJM** relies on the R-packages `survival` and `nnet`, which are hosted on CRAN.

# Installation 
**LatentClassJM** can be installed from Github directly:
```
install.packages("devtools")
library(devtools)
install_github("alexwky/LatentClassJM")
library(LatentClassJM)
```

# Usage
The package contains 2 functions:
Function  | Description
------------- | -------------
LatentClassJM  | Performs the EM algorithm to compute the sieve NPMLE for a semiparametric latent-class joint model
create.data  |  Creates a dataset based on the setting from [Wong et al.](https://doi.org/10.1214/21-AOS2117) (2022)

## LatentClassJM

```
LatentClassJM(Y, X, W, Time, D, ni, Z, G, nknots, knots = NA, degree, covar = "ind",
like.diff1 = FALSE, like.diff2 = TRUE, accelEM = TRUE, bound = 5, qq = 10, epsilon = 0.001, epsilon2 = 1e-06, 
init.param = NULL, qq2 = 10, cal.inf = FALSE, max.iter = 5000, seed = 1)
```
This function performs the (accelerated) EM algorithm to compute the sieve NPMLE. The algorithm starts with the standard EM algorithm. Once the difference between the log-likelihood values or the parameter values of consecutive iterations becomes smaller than a certain threshold, an accelerated EM algorithm (Vardhan and Roland 2008) will be adopted until convergence.

The defines of the model can be found in [Wong et al.](https://doi.org/10.1214/21-AOS2117) (2022) or the help manual with the named "LatentClassJM_0.1.0.pdf".

### Input

The description of the input arguments is as follows:

Argument | Description | Optional | Default 
:----: | ---- | :----: | :----:
Y|An `(n × J × m)` array of longitudinal outcome measurements, where `n` is the sample size, `J` is the number of longitudinal measurement types, and `m` is the maximum number of measurement times. It can contain `NA` values if the number of measurements for a subject is fewer than the maximum number of measurements. The `(i; j; k)`th element corresponds to the `k`th measurement of the `j`th type of longitudinal outcome for the `i`th subject|✖|-
X|An `(n × J × m × pX)` array of covariates (excluding intercept) of the longitudinal outcome model, where `n` is the sample size, `J` is the number of longitudinal measurement types, `m` is the number of measurement times, and `pX` is the number of covariates. The `(i; j; k; l)`th element corresponds to the `l`th covariate for the `k`th measurement of the `j`th type of longitudinal outcome for the `i`th subject|✖|-
W|An `(n × pW)` matrix of covariates for the latent class regression model, where `n` is the sample size, and `pW` is the number of covariate. The `(i; l)`th element corresponds to the `l`th covariate for the `i`th subject|✖|-
Time|An `n`-vector of observed event or censoring times|✖|-
D|An `n`-vector of event indicators|✖|-
ni|An `(n × J)` matrix of numbers of measurements for the longitudinal outcomes|✖|-
Z|An `(n × pZ)` matrix of time-independent covariates for the survival model, where `n` is the sample size, and `pZ` is the number of covariates. The `(i; l)`th element corresponds to the `l`th convariate for the `i`th subject|✖|-
G|Number of latent classes|✖|
nknots|Number of interior knots for the B-spline basis functions|✖|-
knots|An optional vector of interior knot positions. If not supplied, then the interior knots will be selected based on quantiles of the observed event time.|✔|`NA`
degree|The degree of the B-spline basis functions|✖|-
covar|Covariance structure for Y. <ul><li> For covar = `ind`, repeated longitudinal measurements are independent conditional on the random effect $b$ and latent class $C$ $(\sigma_{gj2} = 0)$</li><li> For covar = `exchange`, repeated longitudinal measurements have an exchangeable covariance matrix conditional on the random effect $b$ and latent class $C$ $(\sigma_{gj2} \neq 0)$</li><ul>|✔|`ind`
like.diff1|Logical; If `TRUE`, then convergence of the standard EM algorithm is based on the difference between log-likelihood values of consecutive iterations; otherwise, convergence is based on the maximum difference between parameter values|✔|`FALSE`
like.diff2|Logical; If `TRUE`, then convergence of the accelerated EM algorithm is based on the difference between log-likelihood values of consecutive iterations; otherwise, convergence is based on the maximum difference between parameter values|✔|`FALSE`
accelEM|Logical; The iteration begins with standard EM algorithm. If `TRUE`, then the accelerated EM algorithm will be adopted after the end of the standard EM algorithm; otherwise, the program terminates after the standard EM algorithm|✔|`TRUE`
bound|The upper bound of the absolute value of the parameter estimates|✔|`5`
h|The number of abscissas for the Gauss-Hermite quadrature in the E-step|✔|`10`
h2|The number of abscissas for the Gauss-Hermite quadrature in the calculation of the log-likelihood|✔|`10`
epsilon|Threshold for convergence of standard EM algorithm|✔|`1e-3`
epsilon2|Threshold for convergence of accelerated EM algorithm|✔|`1e-6`
init.param|A named list of user-input initial values of model parameters, including alpha, beta, sigma2, xi, eta, gamma and haz. Please find the format of `init.param` below|✔|`NULL`
cal.inf|Logical; if `TRUE`, then the information matrix will be calculated|✔|`FALSE`
max.iter|Maximum number of iterations|✔|`5000`
seed|Seed used for parameter initialization|✔|`1`

### Example
 
```
dataset <- create.data(n=1000)
result <- LatentClassJM(Y=dataset$Y,X=dataset$X,W=dataset$W,Time=dataset$Time,D=dataset$D,ni=dataset$ni,
Z=dataset$Z,G=4,nknots=2,degree=1,cal.inf=TRUE,init.param=NULL,bound=10,h=20,h2=20,covar="exchange")
```

The input list `init.param` should be in the following format:
Variable | Dimension
:----: | ----
alpha  | A `(G × pW)` matrix
beta  | A `(G × J × pX)` array
sigma2  | <ul><li>If `covar` = `exchange`, then dimension equals `(G × J × 2)`;</li><li> If `covar` = `ind`, then dimension equals `(G × J)`</li></ul>
xi  | A `G`-vector
eta  | A `G`-vector
gamma  | A `(G × q)` matrix, where `q` is the number of B-splines functions; `q` = `nknots`+`degree`+1
haz  | A vector with the same lenght of `Time`
  
### Output
 
The output list consists of the following 14 elements:

Variable  | Description
------------- | -------------
alpha  | A matrix of `(G × pW)` regression parameters for the multinomial regression. The `g`th row is the parameter vector for the `g`th latent class; the last row must be zero
beta  |  A matrix of `(G × J × pX)` regression parameters. The `(g; j)`th row is the `l`th parameter vector for the gth latent class at `j`th measurement type
sigma2  | sigma2 is the variance of the error terms of the longitudinal measurements. <ul><li>If covar = `exchange`, then `sigma2` is a `(G × J × 2)` array. The `(g; j; 1)`th element is $\sigma_{gj1}$, and the `(g; j; 2)`th element is $\sigma_{gj2}$;</li><li> If covar = `ind`, then `sigma2` is a `(G × J)` matrix. The `(g; j)`th element is $\sigma_{gj1}$. In this case, $\sigma_{gj2}$ is fixed to be zero</li>
xi  | A `G`-vector of class-specific variances of the latent variable
gamma  | gamma is a `(G × (pZ + q))` matrix of class-specific regression parameters, consisting of 2 parts, The first `pZ` columns correspond to regression parameters of the covariates `Z`, and the last `q` columns correspond to regression parameters of the spline functions, where `q` = `nknots`+`degree`+1; that is, the `g`th row of gamma is $(\boldsymbol{\gamma}_g^T ; \boldsymbol{\alpha}^T_g )$. The defines can be found in [Wong et al.](https://doi.org/10.1214/21-AOS2117) (2022) or under Details from the help manual `LatentClassJM_0.1.0.pdf`
eta  | A `G`-vector of class-specific regression parameters of the random effect in the survival model
Tt  | A vector of ordered unique observed event times
Haz  | A `(t × q)` matrix of all estimated latent class-specific cumulative hazard function values at `Tt`, where `t` is the length of `Tt`
Bmat  | A `(t × q)` matrix of B-spline basis function values at `Tt`
post.prob  | Subject-specific posterior group probabilities
gridb  | An `(n × G × qq)` array of grid for the adaptive Gauss-Hermite quadrature The `(n; g)`th row corresponds to the grid for the `i`th subject under the `g`th latent class
weightb  | An `(n × G × qq)` array of weight for the adaptive Gauss-Hermite quadrature The `(n; g)`th row corresponds to the weight for the `i`th subject of the `g`th latent class
Information  | Information matrix; `NA` when `cal.inf` = `FALSE`
loglike  | The log-likelihood value


## createdata

```
create.data(n, seed=1)
```
This function is to create a dataset based on the setting from the simulation studies in [Wong et al.](https://doi.org/10.1214/21-AOS2117) (2022) 
 
### Input

The description of the input arguments as below:

Argument | Description | Optional | Default 
:----: | ---- | :----: | :----:
n|Sample size|✖|-
seed|Seed of the random generator|✔|1
 
### Example

```
dataset <- create.data(n=1000)
```
           
### Output
           
The output list consists of following variables:

Variable  | Dimension
------------- | -------------
Y  | n × J × m
X  | n × J × m × pX
W  | n × pW
Time  | n
D  | n
ni | n
Z  | n × pZ

Based on the setting in the simulation studies in [Wong et al.](https://doi.org/10.1214/21-AOS2117) (2022), we fix `J = 2`, `m = 10`, `pX = 3`, `pW = 2`, and `pZ = 2`. 

# Help 

Details about the package can be found in the user manual:
```
?LatentClassJM
```

# Contact 
Wong Kin Yau, Alex <<kin-yau.wong@polyu.edu.hk>>

# Reference 
Vardhan, R. & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. _Scandinavian Journal of Statistics_. **35**: 335–353.

Wong, K. Y., Zeng, D., & Lin, D. Y. (2022). Semiparametric latent-class models for multivariate longitudinal and survival data. _The Annals of Statistics_. **50**: 487–510.
