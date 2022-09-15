#' Sieve nonparametric maximum likelihood estimation for the semiparametric latent-class joint model
#'
#' @description This function performs the (accelerated) EM algorithm to compute the sieve nonparametric maximum likelihood estimator. The algorithm starts with the standard EM algorithm. 
#' Once the difference between the log-likelihood values or the parameter values of consecutive iterations becomes smaller than a certain threshold, an accelerated EM algorithm (Vardhan and Roland 2008) will be adopted until convergence.
#' 
#' @details In this function, we consider a special case of the model introduced in Wong et al. (2022). We consider a model with \eqn{G} latent classes. Let \eqn{C} denote the latent class membership, with \eqn{C = g} if a subject belongs to the \eqn{g}th latent class \eqn{(g = 1, . . . , G)}. We fit a multinomial logistic regression model for \eqn{C}:
#' \deqn{P(C=g \mid \boldsymbol{W}) = \frac{e^{\boldsymbol{\alpha}_{g}^{T}\boldsymbol{W}}}{\sum_{l = 1}^{G}e^{\boldsymbol{\alpha}^{T}_{l}\boldsymbol{W}}},}
#' where \eqn{\boldsymbol{W}} is a vector of time-independent covariates that include the constant 1 and \eqn{\boldsymbol{\alpha}_{g}} is the vector of class-specific regression parameters with \eqn{\boldsymbol{\alpha}_{G} = 0}.
#' Each latent class is characterized by class-specific trajectories of multivariate longitudinal outcomes and a class-specific risk of the event of interest. 
#' The longitudinal outcomes and the event time are assumed to be conditionally independent given the latent class membership and a multivariate random effect.
#' 
#' Suppose that there are \eqn{J} types of longitudinal outcomes, and the \eqn{j}th type is measured at \eqn{N_j} time points. For \eqn{j=1,\dots,J} and \eqn{k=1,\dots,N_j}, 
#' let \eqn{Y_{jk}} denote the \eqn{k}th measurement of the \eqn{j}th longitudinal outcome and \eqn{\boldsymbol{X}_{jk}} denote corresponding covariates, 
#' which include the constant 1. We assume:
#' \deqn{Y_{jk}\mid_{C=g} = \boldsymbol{\beta}^{T}_{g}\boldsymbol{X}_{jk} + b + \epsilon_{jk}}
#' for \eqn{g = 1, \dots,G}, where \eqn{\boldsymbol{X}_{jk}} is a vector of covariates that include the constant 1, \eqn{\boldsymbol{\beta}_g} is a vector of class-specific regression parameters, and \eqn{b} is a normal random effect with mean 0 and variance \eqn{\xi_g}.
#' The error terms \eqn{(\epsilon_{j1},\dots,\epsilon_{jN_{j}})} are dependent zero-mean normal random variables with variance \eqn{\sigma_{gj1}+\sigma_{gj2}} and pairwise covariance \eqn{\sigma_{gj2}}.
#' 
#' Let \eqn{T} denote the event time of interest. We assume a proportional hazards model:
#' \deqn{\lambda(t\mid\boldsymbol{Z}, \boldsymbol{b}, C=g) = \lambda_{g}(t)e^{\boldsymbol{\gamma}_{g}^{T}\boldsymbol{Z} + \eta_{g}b}}
#' where \eqn{\boldsymbol{Z}} is a vector of time-independent covariates, \eqn{\lambda_{g}(.)} is an arbitrary class-specific baseline hazard function, and \eqn{\boldsymbol{\gamma}_{g}} and \eqn{\eta_{g}} are class-specific regression parameters.
#' 
#' We use a sieve nonparametric maximum likelihood estimation method to estimate the model parameters. In particular, we let \eqn{\lambda = \lambda_1} and \eqn{\psi_g = \log(\lambda_g/\lambda_1)} for \eqn{g=1,\dots,G}. We approximate \eqn{\psi_g} by \eqn{\sum_{j=1}^q a_{gj}B_j}, where \eqn{B_1,\dots,B_q} are B-spline functions. Then, we can write the survival model as 
#' \deqn{\lambda(t\mid\boldsymbol{Z},\boldsymbol{b},C=g) = \lambda(t)e^{\boldsymbol{\gamma}_g^T\boldsymbol{Z}+\boldsymbol{a}_g^T\boldsymbol{B}(t)+\eta_{g}b}}
#' where \eqn{\boldsymbol{a}_g = (a_{g1},\dots,a_{gq})^T} and \eqn{\boldsymbol{B}(t) = (B_1(t),\dots,B_q(t))^T}.
#' 
#' @param Y An \eqn{(n \times J \times m)} array of longitudinal outcome measurements, where \eqn{n} is the sample size, \eqn{J} is the number of longitudinal measurement types, and
#' \eqn{m} is the maximum number of measurement times. It can contain \code{NA} values if the number of measurements for a subject is fewer than the maximum number of measurements. The \eqn{(i,j,k)}th element corresponds to the \eqn{k}th measurement of the \eqn{j}th type of longitudinal outcome for the \eqn{i}th subject
#' @param X An \eqn{(n \times J \times m \times p_X)} array of covariates (excluding intercept) of the longitudinal outcome model, 
#' where \eqn{n} is the sample size, \eqn{J} is the number of longitudinal measurement types, \eqn{m} is the number of measurement times, and \eqn{p_X} is the number of covariates. 
#' The \eqn{(i,j,k,l)}th element corresponds to the \eqn{l}th covariate for the \eqn{k}th measurement of the \eqn{j}th type of longitudinal outcome for the \eqn{i}th subject
#' @param W An \eqn{(n \times p_W)} matrix of covariates for the latent class regression model, where \eqn{n} is the sample size, and 
#' \eqn{p_W} is the number of covariate. The \eqn{(i,l)}th element corresponds to the \eqn{l}th covariate for the \eqn{i}th subject
#' @param Time An \eqn{n}-vector of observed event or censoring times
#' @param D An \eqn{n}-vector of event indicators
#' @param ni An \eqn{(n \times J)} matrix of numbers of measurements for the longitudinal outcomes
#' @param Z An \eqn{(n \times p_Z)} matrix of time-independent covariates for the survival model, where \eqn{n} is the sample size, and 
#' \eqn{p_Z} is the number of covariates. The \eqn{(i,l)}th element corresponds to the \eqn{l}th covariate for the \eqn{i}th subject
#' @param G Number of latent classes
#' @param nknots Number of interior knots for the B-spline basis functions
#' @param knots An optional vector of interior knot positions. If not supplied, then the interior knots will be selected based on quantiles of the observed event times
#' @param degree The degree of the B-spline basis functions
#' @param covar Covariance structure for \eqn{Y}. For covar = \code{ind}, repeated longitudinal measurements are independent conditional on the random effect \eqn{b} and latent class \eqn{C} (\eqn{\sigma_{gj2}=0}); 
#' for covar = \code{exchange}, repeated longitudinal measurements have an exchangeable covariance matrix conditional on the random effect \eqn{b} and latent class \eqn{C} (\eqn{\sigma_{gj2}\neq 0}); \eqn{\sigma_{gj2}} is defined in Details below. Default is \code{ind}
#' @param like.diff1 Logical; If \code{TRUE}, then convergence of the standard EM algorithm is based on the difference between log-likelihood values of consecutive iterations; 
#' otherwise, convergence is based on the maximum difference between parameter values; Default is \code{FALSE}
#' @param like.diff2 Logical; If \code{TRUE}, then convergence of the accelerated EM algorithm is based on the difference between log-likelihood values of consecutive iterations; 
#' otherwise, convergence is based on the maximum difference between parameter values; Default is \code{TRUE}
#' @param accelEM Logical; The iteration begins with standard EM algorithm. If \code{TRUE}, then the accelerated EM algorithm will be adopted after the end of the standard EM algorithm; otherwise, the program terminates after the standard EM algorithm 
#' @param bound The upper bound of the absolute value of the parameter estimates
#' @param h The number of abscissas for the Gauss-Hermite quadrature in the E-step
#' @param h2 The number of abscissas for the Gauss-Hermite quadrature in the calculation of the log-likelihood
#' @param epsilon Threshold for convergence of the standard EM algorithm
#' @param epsilon2 Threshold for convergence of the accelerated EM algorithm
#' @param init.param A named list of user-input initial values of model parameters, including \code{alpha}, \code{beta}, \code{sigma2}, \code{xi}, 
#' \code{eta}, \code{gamma} and \code{haz} \itemize{
#' \item \code{alpha} is a matrix of \eqn{(G \times p_W)} regression parameters for the multinomial regression; the last row must be zero
#' \item\code{beta} is an array of \eqn{(G \times J \times p_X)} regression parameters
#' \item \code{sigma2} is the variance of the error terms of the longitudinal measurements. If covar = \code{exchange},
#'   then \code{sigma2} is a \eqn{(G \times J \times 2)} array. The \eqn{(g,j,1)}th element is \eqn{\sigma_{gj1}}, and the \eqn{(g,j,2)}th element is \eqn{\sigma_{gj2}}.
#'   If covar = \code{ind}, then \code{sigma2} is a \eqn{(G \times J)} matrix. The \eqn{(g,j)}th element is \eqn{\sigma_{gj1}}. In this case, \eqn{\sigma_{gj2}} is fixed to be zero
#' \item \code{xi} is a \eqn{G}-vector of class-specific variances of the latent variable
#' \item \code{eta} is a \eqn{G}-vector of class-specific regression parameters of the random effect in the survival model
#' \item \code{gamma} is a \eqn{(G \times (p_Z + q))} matrix of class-specific regression parameters, consisting of 2 parts. The first \eqn{p_Z} columns correspond to regression parameters of the covariates \code{Z}, and the last \eqn{q} columns correspond to regression parameters of the spline functions, where \eqn{q = } \code{nknots}\eqn{+}\code{degree}\eqn{+1}; that is, the \eqn{g}th row of \code{gamma} is \eqn{(\boldsymbol{\gamma}_g^T,\boldsymbol{a}_g^T)}. See Details below
#' \item \code{haz} is a vector of jumps of the first class-specific cumulative hazard function. The jumps should correspond to the ordered unique observed event times
#' }
#' @param cal.inf Logical; if \code{TRUE}, then the information matrix will be calculated
#' @param max.iter Maximum number of iterations
#' @param seed Seed used for parameter initialization; default is 1
#' @return A list of the following components: \itemize{
#'   \item \strong{alpha} : A matrix of \eqn{(G \times p_W)} regression parameters for the multinomial regression. 
#'   The \eqn{g}th row is the parameter vector for the \eqn{g}th latent class; the last row must be zero
#'   \item \strong{beta} : An array of \eqn{(G \times J \times p_X)} regression parameters. 
#'   The \eqn{(g,j)}th row is the \eqn{l}th parameter vector for the \eqn{g}th latent class at \eqn{j}th measurement type
#'   \item \strong{sigma2} : The variance of the error terms of the longitudinal measurements. If covar = \code{exchange}, then \code{sigma2} is a \eqn{(G \times J \times 2)} array. The \eqn{(g,j,1)}th element is \eqn{\sigma_{gj1}}, and the \eqn{(g,j,2)}th element is \eqn{\sigma_{gj2}}. If covar = \code{ind}, then \code{sigma2} is a \eqn{(G \times J)} matrix. The \eqn{(g,j)}th element is \eqn{\sigma_{gj1}}; in this case, \eqn{\sigma_{gj2}} is fixed to be zero
#'   \item \strong{xi} : A \eqn{G}-vector of class-specific variances of the latent variable
#'   \item \strong{gamma} : A \eqn{(G \times (p_Z + q))} matrix of class-specific regression parameters, consisting of 2 parts. The first \eqn{p_Z} columns correspond to regression parameters of the covariates \code{Z}, and the last \eqn{q} columns correspond to regression parameters of the spline functions, where \eqn{q = } \code{nknots}\eqn{+}\code{degree}\eqn{+1}; that is, the \eqn{g}th row of \code{gamma} is \eqn{(\boldsymbol{\gamma}_g^T,\boldsymbol{a}_g^T)}
#'   \item \strong{eta} : A \eqn{G}-vector of class-specific regression parameters of the random effect in the survival model
#'   \item \strong{Tt} : A vector of ordered unique observed event times
#'   \item \strong{Haz} : A \eqn{(t \times q)} matrix of all estimated class-specific cumulative hazard function values at \code{Tt}, where \eqn{t} is the length of \code{Tt}
#'   \item \strong{Bmat} : A \eqn{(t \times q)} matrix of B-spline basis function values at \code{Tt}
#'   \item \strong{post.prob} : Subject-specific posterior group probabilities
#'   \item \strong{gridb} : An \eqn{(n \times G \times h)} array of grid for the adaptive Gauss-Hermite quadrature.
#'   The \eqn{(n,g)}th row corresponds to the grid for the \eqn{i}th subject under the \eqn{g}th latent class
#'   \item \strong{weightb} : An \eqn{(n \times G \times h)} array of weight for the adaptive Gauss-Hermite quadrature.
#'   The \eqn{(n,g)}th row corresponds to the weight for the \eqn{i}th subject of the \eqn{g}th latent class
#'   \item \strong{Information} : Information matrix; \code{NA} when \code{cal.inf} \eqn{=} \code{FALSE}
#'   \item \strong{loglike} : The log-likelihood value
#'   }
#' @author Kin Yau (Alex) Wong <kin-yau.wong@polyu.edu.hk>
#' @references Varadhan, R. & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. Scandinavian Journal of Statistics. 35 335–353. 
#' 
#' Wong, K. Y., Zeng, D., & Lin, D. Y. (2022). Semiparametric latent-class models for multivariate longitudinal and survival data. The Annals of Statistics. 50 487–510. 
#' @seealso \code{survival}
#' @examples 
#' dataset <- create.data(n=1000)
#' result <- LatentClassJM(Y=dataset$Y,X=dataset$X,W=dataset$W,Time=dataset$Time,D=dataset$D,ni=dataset$ni,
#' Z=dataset$Z,G=4,nknots=2,degree=1,cal.inf=TRUE,init.param=NULL,bound=10,h=20,h2=20,covar="exchange")
#' @export
#' @import survival


LatentClassJM <- function(Y,X,W,Time,D,ni,Z,G,nknots,knots=NA,degree,covar="ind",like.diff1=FALSE,like.diff2=TRUE,accelEM=TRUE,bound=5,h=10,epsilon=1e-3,epsilon2=1e-6,init.param=NULL,h2=10,cal.inf=FALSE, max.iter=5000,seed=1) {
  
  loglike <- function(qq.2=h2) {
    if (G > 1) {
      eWa <- matrix(nrow=n,ncol=G)
      for (g in 1:G) eWa[,g] <- exp(W%*%as.vector(alpha[g,]))
      probG <- t(apply(eWa,1,function(x)x/sum(x)))
    } else probG <- matrix(1,nrow=n,ncol=1)
    
    llike <- 0
    gq2 <- gauss.quad(qq.2,kind="hermite")
    gridb2 <- array(dim=c(n,G,qq.2))
    weightb2 <- array(dim=c(n,G,qq.2))
    pt <- (pZ.tilde+1):pZ
    ph <- 1:pZ.tilde
    cumhaz <- matrix(0,n,G)
    for (g in 1:G) {
      cumhaz[1,g] <- exp(sum(Bmat[upto[1],]*gamma[g,pt])) * haz[upto[1]]
      for (i in 2:n) {
        if (D[i]==0 | is.tie[i]) cumhaz[i,g] <- cumhaz[i-1,g] else cumhaz[i,g] <- cumhaz[i-1,g] + exp(sum(Bmat[upto[i],]*gamma[g,pt])) * haz[upto[i]]
      }
    }
    for (i in 1:n) {
      fg <- rep(NA,G)
      for (g in 1:G) {
        cumhaz.i <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph]))
        b <- 0
        for (iter in 1:100) {
          score <- 0
          inf <- 0
          for (k in 1:K) {
            if (ni[i,k] > 0) {
              if (covar=="ind") {
                for (j in 1:ni[i,k]) {
                  score <- score + (Y[i,k,j] - sum(beta[g,k,]*X[i,k,j,]) - b)/sigma2[g,k]
                  inf <- inf - 1/sigma2[g,k]
                }
              } else {
                longY <- Y[i,k,1:ni[i,k]]
                longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
                longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
                score <- score + sum(solve(longSigma,longY-longX%*%beta[g,k,]-b[1]))
                inf <- inf - sum(solve(longSigma))
              }
            }
          }
          score <- score - b/xi[g] + D[i]*eta[g] - eta[g]*exp(eta[g]*b)*cumhaz.i
          inf <- inf - 1/xi[g] - eta[g]^2*exp(eta[g]*b)*cumhaz.i
          b <- b - score / inf
        }
        gridb2[i,g,] <- gq2$nodes/sqrt(-inf) + b
        fb <- rep(0,qq.2)
        fb[] <- 1
        for (k in 1:K) {
          if (ni[i,k] > 0) {
            if (covar=="ind") {
              for (j in 1:ni[i,k]) {
                fb <- fb * sigma2[g,k]^(-0.5) * exp(-(Y[i,k,j] - sum(beta[g,k,]*X[i,k,j,]) - gridb2[i,g,])^2/(2*sigma2[g,k]))
              }
            } else {
              longY <- Y[i,k,1:ni[i,k]]
              longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
              longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
              for (q in 1:qq.2) fb[q] <- fb[q] * det(longSigma)^-0.5 * exp(-0.5 * t(longY - longX%*%beta[g,k,] - gridb2[i,g,q]) %*% solve(longSigma,longY - longX%*%beta[g,k,] - gridb2[i,g,q]))
            }
          }
        }
        fb <- fb * xi[g]^(-0.5) * exp(-gridb2[i,g,]^2/(2*xi[g]))
        if (D[i]==1) fb <- fb * exp(sum(c(baseZ[i,],Bmat[upto[i],])*gamma[g,])+eta[g]*gridb2[i,g,]) * haz[upto[i]]
        fb <- fb * exp(-exp(eta[g]*gridb2[i,g,])*cumhaz.i)
        fg[g] <- sum(fb * gq2$weight * exp(gq2$nodes^2)) / sqrt(-inf)
      }
      fg <- fg * probG[i,]
      llike <- llike + log(sum(fg))
    }
    llike
  }
  
  calInf <- function() {
    
    eWa <- matrix(nrow=n,ncol=G)
    for (g in 1:G) eWa[,g] <- exp(W%*%as.vector(alpha[g,]))
    probG <- t(apply(eWa,1,function(x)x/sum(x)))
    score.alpha <- rep(0,(pW+1)*(G-1))
    score.beta <- rep(0,(pX+1)*K*G)
    score.sigma <- rep(0,G*K*ns)
    score.xi <- rep(0,G)
    score.gamma <- rep(0,pZ*G-(pZ-pZ.tilde))
    score.eta <- rep(0,G)
    score.lambda <- rep(0,length(Tt))
    size <- length(c(score.alpha,score.beta,score.sigma,score.xi,score.gamma,score.eta,score.lambda))
    Inf1 <- matrix(0,size,size)
    
    pt <- (pZ.tilde+1):pZ
    ph <- 1:pZ.tilde
    cumhaz <- matrix(0,n,G)
    cumhaz.B <- array(0,dim=c(n,G,ncol(Bmat)))
    cumhaz.B2 <- array(0,dim=c(n,G,ncol(Bmat),ncol(Bmat)))
    for (g in 1:G) {
      cumhaz[1,g] <- exp(sum(Bmat[upto[1],]*gamma[g,pt])) * haz[upto[1]]
      cumhaz.B[1,g,] <- exp(sum(Bmat[upto[1],]*gamma[g,pt])) * haz[upto[1]] * Bmat[upto[1],]
      cumhaz.B2[1,g,,] <- exp(sum(Bmat[upto[1],]*gamma[g,pt])) * haz[upto[1]] * Bmat[upto[1],] %*% t(Bmat[upto[1],])
      for (i in 2:n) {
        if (D[i]==0 | is.tie[i]) {
          cumhaz[i,g] <- cumhaz[i-1,g]
          cumhaz.B[i,g,] <- cumhaz.B[i-1,g,]
          cumhaz.B2[i,g,,] <- cumhaz.B2[i-1,g,,]
        } else {
          cumhaz[i,g] <- cumhaz[i-1,g] + exp(sum(Bmat[upto[i],]*gamma[g,pt])) * haz[upto[i]]
          cumhaz.B[i,g,] <- cumhaz.B[i-1,g,] + exp(sum(Bmat[upto[i],]*gamma[g,pt])) * haz[upto[i]] * Bmat[upto[i],]
          cumhaz.B2[i,g,,] <- cumhaz.B2[i-1,g,,] + exp(sum(Bmat[upto[i],]*gamma[g,pt])) * haz[upto[i]] * Bmat[upto[i],] %*% t(Bmat[upto[i],])
        }
      }
    }
    for (i in 1:n) {
      S0 <- rep(0,G)
      S1 <- array(0,dim=c(G,pZ))
      for (g in 1:G) {
        S0[g] <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph]))
        S1[g,ph] <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph])) * baseZ[i,]
        S1[g,pt] <- cumhaz.B[i,g,] * exp(sum(baseZ[i,]*gamma[g,ph]))
      }
      Inf1i <- matrix(0,size,size)
      U1i <- rep(0,size)
      for (g in 1:G) {
        for (m in 1:h) {
          score.alpha[] <- 0
          score.beta[] <- 0
          score.sigma[] <- 0
          score.xi[] <- 0
          score.gamma[] <- 0
          score.eta[] <- 0
          score.lambda[] <- 0
          if (g < G) score.alpha[((g-1)*(pW+1)+1):(g*(pW+1))] <- 1
          score.alpha <- score.alpha - rep(probG[i,-G],each=pW+1)
          score.alpha <- score.alpha * rep(W[i,],times=G-1)
          for (k in 1:K) {
            r1 <- ((g-1)*K*(pX+1)+(k-1)*(pX+1)+1):((g-1)*K*(pX+1)+k*(pX+1))
            r2 <- ((g-1)*K*ns+(k-1)*ns+1):((g-1)*K*ns+(k-1)*ns+ns)
            if (ni[i,k] > 0) {
              if (covar=="ind") {
                for (j in 1:ni[i,k]) {
                  score.beta[r1] <- score.beta[r1] + (Y[i,k,j]-sum(X[i,k,j,]*beta[g,k,])-gridb[i,g,m])*X[i,k,j,]/sigma2[g,k]
                  score.sigma[r2] <- score.sigma[r2] - 1/(2*sigma2[g,k]) + (Y[i,k,j]-sum(X[i,k,j,]*beta[g,k,])-gridb[i,g,m])^2/(2*sigma2[g,k]^2)
                }
              } else {
                longY <- Y[i,k,1:ni[i,k]]
                longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
                longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
                score.beta[r1] <- score.beta[r1] + t(longY-longX%*%beta[g,k,]-gridb[i,g,m])%*%solve(longSigma,longX)
                score.sigma[r2[1]] <- score.sigma[r2[1]] - 0.5*sum(diag(solve(longSigma))) + 0.5*sum((longY-longX%*%beta[g,k,]-gridb[i,g,m])^2)/sigma2[g,k,1]^2 -
                  0.5*sigma2[g,k,2]*(2*sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])/(sigma2[g,k,1]^2+ni[i,k]*sigma2[g,k,2]*sigma2[g,k,1])^2*(sum(longY-longX%*%beta[g,k,]-gridb[i,g,m]))^2
                score.sigma[r2[2]] <- score.sigma[r2[2]] - 0.5*ni[i,k]/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2]) + 0.5*(sum(longY-longX%*%beta[g,k,]-gridb[i,g,m]))^2 / (sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2
              }
            }
          }
          score.xi[g] <- -1/(2*xi[g]) + gridb[i,g,m]^2 / (2*xi[g]^2)
          if (g == 1) {
            r3 <- 1:pZ.tilde
            score.gamma[r3] <- D[i]*baseZ[i,] - S1[g,1:pZ.tilde]*exp(gridb[i,g,m]*eta[g])
          } else {
            r3 <- ((g-1)*pZ+1):(g*pZ)-(pZ-pZ.tilde)
            score.gamma[r3] <- D[i]*c(baseZ[i,],Bmat[upto[i],]) - S1[g,]*exp(gridb[i,g,m]*eta[g])
          }
          score.eta[g] <- D[i]*gridb[i,g,m] - S0[g]*exp(gridb[i,g,m]*eta[g])*gridb[i,g,m]
          for (j in 1:upto[i]) score.lambda[j] <- D[i]/haz[j] - exp(sum(c(baseZ[i,],Bmat[j,])*gamma[g,])+eta[g]*gridb[i,g,m])
          all.score <- c(score.alpha,score.beta,score.sigma,score.xi,score.gamma,score.eta,score.lambda)
          U1i <- U1i + all.score * fg[i,g] * weightb[i,g,m]
          Inf1i <- Inf1i + all.score %*% t(all.score) * fg[i,g] * weightb[i,g,m]
        }
      }
      Inf1 <- Inf1 + (Inf1i - U1i %*% t(U1i))
    }
    
    Sigma.alpha <- matrix(0,ncol=(G-1)*(pW+1),nrow=(G-1)*(pW+1))
    if (G > 1) {
      for (g1 in 1:(G-1)) {
        for (g2 in 1:(G-1)) {
          r1 <- ((g1-1)*(pW+1)+1):(g1*(pW+1))
          r2 <- ((g2-1)*(pW+1)+1):(g2*(pW+1))
          for (i in 1:n) {
            if (g1==g2) Sigma.alpha[r1,r2] <- Sigma.alpha[r1,r2] - probG[i,g1]*(1-probG[i,g2])*W[i,]%*%t(W[i,]) else
              Sigma.alpha[r1,r2] <- Sigma.alpha[r1,r2] + probG[i,g1]*probG[i,g2]*W[i,]%*%t(W[i,])
          }
        }
      }
    }
    Sigma.beta <- matrix(0,nrow=G*K*(pX+1),ncol=G*K*(pX+1))
    for (g in 1:G) {
      for (k in 1:K) {
        r1 <- ((g-1)*K*(pX+1)+(k-1)*(pX+1)+1):((g-1)*K*(pX+1)+k*(pX+1))
        for (i in 1:n) {
          if (ni[i,k] > 0) {
            if (covar=="ind") {
              for (j in 1:ni[i,k]) {
                Sigma.beta[r1,r1] <- Sigma.beta[r1,r1] - fg[i,g] * X[i,k,j,] %*% t(X[i,k,j,]) / sigma2[g,k]
              }
            } else {
              longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
              longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
              Sigma.beta[r1,r1] <- Sigma.beta[r1,r1] - fg[i,g] * t(longX) %*% solve(longSigma,longX)
            }
          }
        }
      }
    }
    Sigma.sigma <- matrix(0,nrow=G*K*ns,ncol=G*K*ns)
    for (g in 1:G) {
      for (k in 1:K) {
        r2 <- ((g-1)*K*ns+(k-1)*ns+1):((g-1)*K*ns+(k-1)*ns+ns)
        for (i in 1:n) {
          if (covar=="ind") {
            Sigma.sigma[r2,r2] <- Sigma.sigma[r2,r2] - fg[i,g] * ni[i,k] / (2*sigma2[g,k]^2)
          } else {
            Sigma.sigma[r2[1],r2[1]] <- Sigma.sigma[r2[1],r2[1]] - fg[i,g]*ni[i,k]*((sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2-sigma2[g,k,2]*(2*sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])) / (2*sigma2[g,k,1]^2*(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2)
            Sigma.sigma[r2[1],r2[2]] <- Sigma.sigma[r2[1],r2[2]] - fg[i,g]*ni[i,k]/(2*(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2)
            Sigma.sigma[r2[2],r2[1]] <- Sigma.sigma[r2[1],r2[2]]
            Sigma.sigma[r2[2],r2[2]] <- Sigma.sigma[r2[2],r2[2]] - fg[i,g]*ni[i,k]^2/(2*(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2)
          }
        }
      }
    }
    Sigma.xi <- matrix(0,nrow=G,ncol=G)
    for (g in 1:G) {
      for (i in 1:n) {
        Sigma.xi[g,g] <- Sigma.xi[g,g] - fg[i,g] / (2*xi[g]^2)
      }
    }
    Sigma.gamma.eta <- matrix(0,nrow=(pZ+1)*G,ncol=(pZ+1)*G)
    Sigma.ge.lambda <- matrix(0,nrow=(pZ+1)*G,ncol=length(Tt))
    
    for (i in 1:n) {
      S0 <- rep(0,G)
      S1 <- array(0,dim=c(G,pZ))
      S2 <- array(0,dim=c(G,pZ,pZ))
      for (g in 1:G) {
        S0[g] <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph]))
        S1[g,ph] <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph])) * baseZ[i,]
        S1[g,pt] <- cumhaz.B[i,g,] * exp(sum(baseZ[i,]*gamma[g,ph]))
        S2[g,ph,ph] <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph])) * baseZ[i,] %*% t(baseZ[i,])
        S2[g,pt,ph] <- cumhaz.B[i,g,] %*% t(baseZ[i,]) * exp(sum(baseZ[i,]*gamma[g,ph]))
        S2[g,ph,pt] <- baseZ[i,] %*% t(cumhaz.B[i,g,]) * exp(sum(baseZ[i,]*gamma[g,ph]))
        S2[g,pt,pt] <- cumhaz.B2[i,g,,] * exp(sum(baseZ[i,]*gamma[g,ph]))
      }
      for (g in 1:G) {
        for (m in 1:h) {
          r3 <- ((g-1)*pZ+1):(g*pZ)
          r4 <- G*pZ + g
          Sigma.gamma.eta[r3,r3] <- Sigma.gamma.eta[r3,r3] - S2[g,,] * exp(gridb[i,g,m]*eta[g]) * fg[i,g] * weightb[i,g,m]
          Sigma.gamma.eta[r4,r4] <- Sigma.gamma.eta[r4,r4] - S0[g] * exp(gridb[i,g,m]*eta[g]) * gridb[i,g,m]^2 * fg[i,g] * weightb[i,g,m]
          Sigma.gamma.eta[r3,r4] <- Sigma.gamma.eta[r3,r4] - S1[g,] * exp(gridb[i,g,m]*eta[g]) * gridb[i,g,m] * fg[i,g] * weightb[i,g,m]
          Sigma.gamma.eta[r4,r3] <- Sigma.gamma.eta[r4,r3] - S1[g,] * exp(gridb[i,g,m]*eta[g]) * gridb[i,g,m] * fg[i,g] * weightb[i,g,m]
          for (j in 1:upto[i]) {
            Sigma.ge.lambda[r3,j] <- Sigma.ge.lambda[r3,j] - exp(sum(c(baseZ[i,],Bmat[j,])*gamma[g,])+gridb[i,g,m]*eta[g]) * c(baseZ[i,],Bmat[j,]) * fg[i,g] * weightb[i,g,m]
            Sigma.ge.lambda[r4,j] <- Sigma.ge.lambda[r4,j] - exp(sum(c(baseZ[i,],Bmat[j,])*gamma[g,])+gridb[i,g,m]*eta[g]) * gridb[i,g,m] * fg[i,g] * weightb[i,g,m]
          }
        }
      }
    }
    Sigma.gamma.eta <- Sigma.gamma.eta[-((pZ.tilde+1):pZ),-((pZ.tilde+1):pZ)]
    Sigma.ge.lambda <- Sigma.ge.lambda[-((pZ.tilde+1):pZ),]
    Sigma.lambda <- diag(-1/haz^2)
    
    sizes <- c(length(score.alpha),length(score.beta),length(score.sigma),length(score.xi),length(score.gamma)+length(score.eta),length(score.lambda))
    cum.sizes <- cumsum(c(0,sizes))
    Inf2 <- matrix(0,size,size)
    if (G > 1) Inf2[(cum.sizes[1]+1):cum.sizes[2],(cum.sizes[1]+1):cum.sizes[2]] <- Sigma.alpha
    Inf2[(cum.sizes[2]+1):cum.sizes[3],(cum.sizes[2]+1):cum.sizes[3]] <- Sigma.beta
    Inf2[(cum.sizes[3]+1):cum.sizes[4],(cum.sizes[3]+1):cum.sizes[4]] <- Sigma.sigma
    Inf2[(cum.sizes[4]+1):cum.sizes[5],(cum.sizes[4]+1):cum.sizes[5]] <- Sigma.xi
    Inf2[(cum.sizes[5]+1):cum.sizes[6],(cum.sizes[5]+1):cum.sizes[6]] <- Sigma.gamma.eta
    Inf2[(cum.sizes[5]+1):cum.sizes[6],(cum.sizes[6]+1):cum.sizes[7]] <- Sigma.ge.lambda
    Inf2[(cum.sizes[6]+1):cum.sizes[7],(cum.sizes[5]+1):cum.sizes[6]] <- t(Sigma.ge.lambda)
    Inf2[(cum.sizes[6]+1):cum.sizes[7],(cum.sizes[6]+1):cum.sizes[7]] <- Sigma.lambda
    return(-Inf2-Inf1)
  }
  
  EMStep <- function(fg,weightb,gridb) {
    
    #    pen <- 0
    pt <- (pZ.tilde+1):pZ
    ph <- 1:pZ.tilde
    cumhaz <- matrix(0,n,G)
    for (g in 1:G) {
      cumhaz[1,g] <- exp(sum(Bmat[upto[1],]*gamma[g,pt])) * haz[upto[1]]
      for (i in 2:n) {
        if (D[i]==0 | is.tie[i]) cumhaz[i,g] <- cumhaz[i-1,g] else cumhaz[i,g] <- cumhaz[i-1,g] + exp(sum(Bmat[upto[i],]*gamma[g,pt])) * haz[upto[i]]
      }
    }
    fg <- array(dim=c(n,G))
    for (i in 1:n) {
      
      tmp.fg <- rep(0,G)
      all.fb <- matrix(nrow=G,ncol=h)
      all.inf <- rep(0,G)
      for (g in 1:G) {
        cumhaz.i <- cumhaz[i,g] * exp(sum(baseZ[i,]*gamma[g,ph]))
        b <- 0
        diff.b <- 10
        for (iter in 1:100) {
          old.b <- b
          score <- 0
          inf <- 0
          for (k in 1:K) {
            if (ni[i,k] > 0) {
              if (covar=="ind") {
                for (j in 1:ni[i,k]) {
                  score <- score + (Y[i,k,j] - sum(beta[g,k,]*X[i,k,j,]) - b)/sigma2[g,k]
                  inf <- inf - 1/sigma2[g,k]
                }
              } else {
                longY <- Y[i,k,1:ni[i,k]]
                longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
                longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
                score <- score + sum(solve(longSigma,longY-longX%*%beta[g,k,]-b[1]))
                inf <- inf - sum(solve(longSigma))
              }
            }
          }
          score <- score - b/xi[g] + D[i]*eta[g] - eta[g]*exp(eta[g]*b)*cumhaz.i
          inf <- inf - 1/xi[g] - eta[g]^2*exp(eta[g]*b)*cumhaz.i
          b <- b - score / inf
          diff.b <- abs(b-old.b)
          if (diff.b < 1e-3) break
        }
        gridb[i,g,] <- gq$nodes/sqrt(-inf) + b
        fb <- rep(0,h)
        fb[] <- sum(W[i,]*alpha[g,])
        for (k in 1:K) {
          if (ni[i,k] > 0) {
            if (covar=="ind") {
              for (j in 1:ni[i,k]) {
                fb <- fb - 0.5 * log(sigma2[g,k]) - (Y[i,k,j] - sum(beta[g,k,]*X[i,k,j,]) - gridb[i,g,])^2/(2*sigma2[g,k])
              }
            } else {
              longY <- Y[i,k,1:ni[i,k]]
              longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
              longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
              for (q in 1:h) {
                fb[q] <- fb[q] - 0.5 * log(det(longSigma)) - 0.5 * t(longY - longX%*%beta[g,k,] - gridb[i,g,q]) %*% solve(longSigma,longY-longX%*%beta[g,k,]-gridb[i,g,q])
              }
            }
          }
        }
        fb <- fb - 0.5 * log(xi[g]) - gridb[i,g,]^2/(2*xi[g])
        if (D[i]==1) fb <- fb + sum(c(baseZ[i,],Bmat[upto[i],])*gamma[g,])+eta[g]*gridb[i,g,]
        fb <- fb - exp(eta[g]*gridb[i,g,])*cumhaz.i
        all.fb[g,] <- fb
        all.inf[g] <- inf
      }
      gpmax <- apply(all.fb,1,max)
      for (g in 1:G) {
        weightb[i,g,] <- exp(all.fb[g,]-gpmax[g]) * gq$weight * exp(gq$nodes^2)
        tmp.fg[g] <- sum(exp(all.fb[g,]-gpmax[g]) * gq$weight * exp(gq$nodes^2)) / sqrt(-all.inf[g])
      }
      for (g in 1:G) weightb[i,g,] <- weightb[i,g,] / sum(weightb[i,g,])
      fg[i,] <- tmp.fg * exp(gpmax-max(gpmax)) / sum(tmp.fg * exp(gpmax-max(gpmax)))
      
    }
    
    if (G > 1) {
      eWa <- matrix(nrow=n,ncol=G)
      for (g in 1:G) eWa[,g] <- exp(W%*%as.vector(alpha[g,]))
      probG <- t(apply(eWa,1,function(x)x/sum(x)))
      Ua <- matrix(0,nrow=pW+1,ncol=G-1)
      for (g in 1:(G-1)) {
        Ua[,g] <- colSums((fg[,g]-probG[,g]) * W)
      }
      Ua <- as.vector(Ua)
      Ia <- matrix(0,ncol=(G-1)*(pW+1),nrow=(G-1)*(pW+1))
      for (g1 in 1:(G-1)) {
        for (g2 in 1:(G-1)) {
          r1 <- ((g1-1)*(pW+1)+1):(g1*(pW+1))
          r2 <- ((g2-1)*(pW+1)+1):(g2*(pW+1))
          for (i in 1:n) {
            if (g1==g2) Ia[r1,r2] <- Ia[r1,r2] - probG[i,g1]*(1-probG[i,g2])*W[i,]%*%t(W[i,]) else
              Ia[r1,r2] <- Ia[r1,r2] + probG[i,g1]*probG[i,g2]*W[i,]%*%t(W[i,])
          }
        }
      }
      tmp.alpha.old <- as.vector(t(alpha[-G,,drop=FALSE]))
      tmp.alpha <- tmp.alpha.old - solve(Ia,Ua)
      alpha <- rbind(matrix(tmp.alpha,nrow=G-1,ncol=pW+1,byrow=TRUE),rep(0,pW+1))
      if (any(abs(alpha) > bound)) {
        step <- - solve(Ia,Ua)
        scale <- min(abs(c(((bound - tmp.alpha.old) / step)[step > 0], ((tmp.alpha.old + bound) / step)[step < 0])))
        scale <- min(scale,1)
        tmp.alpha <- tmp.alpha.old + scale * step
        alpha <- rbind(matrix(tmp.alpha,nrow=G-1,ncol=pW+1,byrow=TRUE),rep(0,pW+1))
        print("alpha bound enforced")
      }
    } else {
      probG <- rep(1,length=n)
      alpha[,] <- 0
    }
    Eb <- array(dim=c(n,G))
    Eb2 <- array(dim=c(n,G))
    Eexpeb <- array(dim=c(n,G))
    Eexpebb <- array(dim=c(n,G))
    Eexpebb2 <- array(dim=c(n,G))
    for (g in 1:G) {
      Eb[,g] <- rowSums(gridb[,g,] * weightb[,g,])
      Eb2[,g] <- rowSums(gridb[,g,]^2 * weightb[,g,])
      Eexpeb[,g] <- rowSums(exp(t(t(gridb[,g,])*eta[g])) * weightb[,g,])
      Eexpebb[,g] <- rowSums(exp(t(t(gridb[,g,])*eta[g]))*gridb[,g,] * weightb[,g,])
      Eexpebb2[,g] <- rowSums(exp(t(t(gridb[,g,])*eta[g]))*gridb[,g,]^2 * weightb[,g,])
    }
    
    for (g in 1:G) {
      for (k in 1:K) {
        XXt <- matrix(0,nrow=pX+1,ncol=pX+1)
        XY <- rep(0,pX+1)
        for (i in 1:n) {
          if (ni[i,k] > 0) {
            if (covar=="ind") {
              for (j in 1:ni[i,k]) {
                XXt <- XXt + fg[i,g] * X[i,k,j,]%*%t(X[i,k,j,])
                XY <- XY + fg[i,g] * (Y[i,k,j] - Eb[i,g]) * X[i,k,j,]
              }
            } else {
              longY <- Y[i,k,1:ni[i,k]]
              longX <- X[i,k,,][1:ni[i,k],,drop=FALSE]
              longSigma <- diag(sigma2[g,k,1],ni[i,k]) + matrix(sigma2[g,k,2],ni[i,k],ni[i,k])
              XXt <- XXt + fg[i,g] * t(longX) %*% solve(longSigma,longX)
              XY <- XY + fg[i,g] * t(longX) %*% solve(longSigma,longY - Eb[i,g])
            }
          }
        }
        beta[g,k,] <- solve(XXt,XY)
        beta[g,k,beta[g,k,] > bound] <- bound
        beta[g,k,beta[g,k,] < -bound] <- -bound
        if (covar=="ind") {
          tmpS <- ((Y[,k,] - apply(sweep(X[,k,,],3,beta[g,k,],FUN="*"),c(1,2),sum) - matrix(Eb[,g],nrow=n,ncol=maxni))^2 +
                     matrix(Eb2[,g] - Eb[,g]^2,nrow=n,ncol=maxni)) * fg[,g]
          s <- sum(tmpS[!is.na(ind.mat[[k]])])
          sigma2[g,k] <- s / sum(ni[,k] * fg[,g])
          if (sigma2[g,k] > bound) sigma2[g,k] <- bound
        } else {
          oldsigma <- sigma2[g,k,]
          for (it in 1:20) {
            s.sigma <- rep(0,2)
            inf.sigma <- matrix(0,2,2)
            for (i in 1:n) {
              if (ni[i,k]>0) {
                yy <- 0
                y2 <- 0
                sumy <- 0
                for (j in 1:ni[i,k]) {
                  yy <- yy + (Y[i,k,j] - sum(X[i,k,j,]*beta[g,k,]) - Eb[i,g])^2 + Eb2[i,g] - Eb[i,g]^2
                  sumy <- sumy + Y[i,k,j] - sum(X[i,k,j,]*beta[g,k,])
                }
                y2 <- sumy^2 - 2*sumy*ni[i,k]*Eb[i,g] + ni[i,k]^2*Eb2[i,g]
                s.sigma[1] <- s.sigma[1] + fg[i,g] * (-0.5*ni[i,k]*(sigma2[g,k,1]+(ni[i,k]-1)*sigma2[g,k,2])/(sigma2[g,k,1]*(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2]))+
                                                        0.5*yy/sigma2[g,k,1]^2-0.5*y2*sigma2[g,k,2]*(2*sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])/(sigma2[g,k,1]^2+ni[i,k]*sigma2[g,k,2]*sigma2[g,k,1])^2)
                s.sigma[2] <- s.sigma[2] + fg[i,g] * (-0.5*ni[i,k]/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2]) + 0.5*y2/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2)
                inf.sigma[1,1] <- inf.sigma[1,1] + fg[i,g] * (0.5*ni[i,k]*(sigma2[g,k,1]^2+2*sigma2[g,k,1]*sigma2[g,k,2]*(ni[i,k]-1)+ni[i,k]*(ni[i,k]-1)*sigma2[g,k,2]^2)/(sigma2[g,k,1]^2+ni[i,k]*sigma2[g,k,1]*sigma2[g,k,2])^2-
                                                                yy/sigma2[g,k,1]^3+y2*(3*sigma2[g,k,2]*sigma2[g,k,1]^2+3*ni[i,k]*sigma2[g,k,1]*sigma2[g,k,2]^2+ni[i,k]^2*sigma2[g,k,2]^3)/(sigma2[g,k,1]^2+ni[i,k]*sigma2[g,k,1]*sigma2[g,k,2])^3)
                inf.sigma[1,2] <- inf.sigma[1,2] + fg[i,g] * (0.5*ni[i,k]/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2 - y2/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^3)
                inf.sigma[2,2] <- inf.sigma[2,2] + fg[i,g] * (0.5*ni[i,k]^2/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^2 - y2*ni[i,k]/(sigma2[g,k,1]+ni[i,k]*sigma2[g,k,2])^3)
              }
            }
            inf.sigma[2,1] <- inf.sigma[1,2]
            inf.sigma <- inf.sigma * sigma2[g,k,]%*%t(sigma2[g,k,]) + diag(s.sigma * sigma2[g,k,])
            s.sigma <- s.sigma * sigma2[g,k,]
            sigma2gk <- log(sigma2[g,k,])
            sigma2gk <- exp(sigma2gk - solve(inf.sigma,s.sigma))
            oldsigma <- sigma2[g,k,]
            sigma2[g,k,] <- sigma2gk
            diff <- sum((oldsigma - sigma2[g,k,])^2)
            #		print(c(it,s.sigma,sigma2[g,k,]))
            if (any(abs(log(sigma2[g,k,])) > bound)) {
              step <- - solve(inf.sigma,s.sigma)
              scale <- min(abs(c(((bound - log(oldsigma)) / step)[step > 0], ((log(oldsigma) + bound) / step)[step < 0])))
              scale <- min(scale,1)
              sigma2[g,k,] <- exp(log(oldsigma) + scale * step)
              print("sigma bound enforced")
            }
            if (diff < 1e-2) break
          }
          if (sigma2[g,k,1] > bound) sigma2[g,k,1] <- bound
          if (sigma2[g,k,2] > bound) sigma2[g,k,2] <- bound
        }
        #	print(s.sigma)
        #	print(sigma2[g,k,])
      }
    }
    if (any(abs(beta) == bound)) print("beta bound enforced")
    if (any(abs(sigma2) == bound)) print("sigma bound enforced")
    
    for (g in 1:G) {
      xi[g] <- sum(fg[,g] * Eb2[,g]) / sum(fg[,g])
      if (xi[g] > bound) {
        xi[g] <- bound
        print("xi bound enforced")
      }
    }
    
    S0 <- rep(0,n)
    S1 <- array(0,dim=c(n,g,pZ+1))
    S2 <- array(0,dim=c(n,g,pZ+1,pZ+1))
    for (g in 1:G) {
      ebZg <- exp(baseZ%*%gamma[g,ph])
      S0g <- rep(0,n)
      sumS0 <- sum(fg[,g] * ebZg * Eexpeb[,g])
      sumS1.1 <- colSums(baseZ * as.vector(fg[,g] * ebZg * Eexpeb[,g]))
      sumS1.2 <- sum(fg[,g] * ebZg * Eexpebb[,g])
      sumS2.11 <- matrix(0,pZ.tilde,pZ.tilde)
      for (i in 1:n) sumS2.11 <- sumS2.11 + fg[i,g] * ebZg[i] * Eexpeb[i,g] * baseZ[i,] %*% t(baseZ[i,])
      sumS2.12 <- colSums(baseZ * as.vector(fg[,g] * ebZg * Eexpebb[,g]))
      sumS2.22 <- sum(fg[,g] * ebZg * Eexpebb2[,g])
      
      S0g[1] <- sumS0
      S1[1,g,ph] <- sumS1.1
      S1[1,g,pt] <- sumS0
      S1[1,g,pZ+1] <- sumS1.2
      S2[1,g,ph,ph] <- sumS2.11
      for (tt in pt) S2[1,g,ph,tt] <- t(sumS1.1)
      S2[1,g,ph,pZ+1] <- sumS2.12
      for (tt in pt) S2[1,g,tt,ph] <- sumS1.1
      S2[1,g,pt,pt] <- sumS0
      S2[1,g,pt,pZ+1] <- sumS1.2
      S2[1,g,pZ+1,ph] <- sumS2.12
      S2[1,g,pZ+1,pt] <- sumS1.2
      S2[1,g,pZ+1,pZ+1] <- sumS2.22
      accum0 <- 0
      accum1 <- rep(0,pZ+1)
      accum2 <- matrix(0,pZ+1,pZ+1)
      for (i in 2:n) {
        accum0 <- accum0 + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g]
        accum1[ph] <- accum1[ph] + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g] * baseZ[i-1,]
        accum1[pt] <- accum1[pt] + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g]
        accum1[pZ+1] <- accum1[pZ+1] + fg[i-1,g] * ebZg[i-1] * Eexpebb[i-1,g]
        accum2[ph,ph] <- accum2[ph,ph] + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g] * baseZ[i-1,] %*% t(baseZ[i-1,])
        for (tt in pt) accum2[ph,tt] <- accum2[ph,tt] + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g] * baseZ[i-1,]
        accum2[ph,pZ+1] <- accum2[ph,pZ+1] + fg[i-1,g] * ebZg[i-1] * Eexpebb[i-1,g] * baseZ[i-1,]
        for (tt in pt) accum2[tt,ph] <- accum2[tt,ph] + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g] * baseZ[i-1,]
        accum2[pt,pt] <- accum2[pt,pt] + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g]
        accum2[pt,pZ+1] <- accum2[pt,pZ+1] + fg[i-1,g] * ebZg[i-1] * Eexpebb[i-1,g]
        accum2[pZ+1,ph] <- accum2[pZ+1,ph] + fg[i-1,g] * ebZg[i-1] * Eexpebb[i-1,g] * baseZ[i-1,]
        accum2[pZ+1,pt] <- accum2[pZ+1,pt] + fg[i-1,g] * ebZg[i-1] * Eexpebb[i-1,g]
        accum2[pZ+1,pZ+1] <- accum2[pZ+1,pZ+1] + fg[i-1,g] * ebZg[i-1] * Eexpebb2[i-1,g]
        if (D[i]==1 & !is.tie[i]) {
          S0g[i] <- S0g[i-1] - accum0
          S1[i,g,ph] <- S1[i-1,g,ph] - accum1[ph]
          S1[i,g,pt] <- S1[i-1,g,pt] - accum1[pt]
          S1[i,g,pZ+1] <- S1[i-1,g,pZ+1] - accum1[pZ+1]
          
          S2[i,g,ph,ph] <- S2[i-1,g,ph,ph] - accum2[ph,ph]
          for (tt in pt) S2[i,g,ph,tt] <- S2[i-1,g,ph,tt] - accum2[ph,tt]
          S2[i,g,ph,pZ+1] <- S2[i-1,g,ph,pZ+1] - accum2[ph,pZ+1]
          for (tt in pt) S2[i,g,tt,ph] <- S2[i-1,g,tt,ph] - accum2[tt,ph]
          S2[i,g,pt,pt] <- S2[i-1,g,pt,pt] - accum2[pt,pt]
          S2[i,g,pt,pZ+1] <- S2[i-1,g,pt,pZ+1] - accum2[pt,pZ+1]
          S2[i,g,pZ+1,ph] <- S2[i-1,g,pZ+1,ph] - accum2[pZ+1,ph]
          S2[i,g,pZ+1,pt] <- S2[i-1,g,pZ+1,pt] - accum2[pZ+1,pt]
          S2[i,g,pZ+1,pZ+1] <- S2[i-1,g,pZ+1,pZ+1] - accum2[pZ+1,pZ+1]
          accum0 <- 0
          accum1 <- rep(0,pZ+1)
          accum2 <- matrix(0,pZ+1,pZ+1)
        } else {
          S0g[i] <- S0g[i-1]
          S1[i,g,] <- S1[i-1,g,]
          S2[i,g,,] <- S2[i-1,g,,]
        }
      }
      eBg <- exp(Bmat%*%gamma[g,pt])
      j <- 0
      for (i in which(D==1)) {
        if (!is.tie[i]) j <- j + 1
        S0g[i] <- S0g[i] * eBg[j]
        S1[i,g,c(ph,pZ+1)] <- S1[i,g,c(ph,pZ+1)] * eBg[j]
        S1[i,g,pt] <- S1[i,g,pt] * eBg[j] * Bmat[j,]
        S2[i,g,c(ph,pZ+1),c(ph,pZ+1)] <- S2[i,g,c(ph,pZ+1),c(ph,pZ+1)] * eBg[j]
        for (tt in c(ph,pZ+1)) S2[i,g,tt,pt] <- S2[i,g,tt,pt] * eBg[j] * Bmat[j,]
        for (tt in c(ph,pZ+1)) S2[i,g,pt,tt] <- S2[i,g,pt,tt] * eBg[j] * Bmat[j,]
        S2[i,g,pt,pt] <- S2[i,g,pt,pt] * eBg[j] * Bmat[j,] %*% t(Bmat[j,])
      }
      S0 <- S0 + S0g
    }
    Ut <- rep(0,G*(pZ+1))
    It <- matrix(0,G*(pZ+1),G*(pZ+1))
    for (g1 in 1:G) {
      r1 <- ((g1-1)*(pZ+1)+1):(g1*(pZ+1))
      Ut[r1] <- colSums(D * (fg[,g1] * cbind(baseZ,Bmat[upto,],Eb[,g1]) - S1[,g1,]/S0))
      for (g2 in 1:G) {
        r2 <- ((g2-1)*(pZ+1)+1):(g2*(pZ+1))
        for (i in 1:n) {
          if (g1==g2) It[r1,r2] <- It[r1,r2] - D[i] * (S2[i,g1,,]/S0[i] - S1[i,g1,]%*%t(S1[i,g2,])/S0[i]^2) else
            It[r1,r2] <- It[r1,r2] + D[i] * S1[i,g1,]%*%t(S1[i,g2,])/S0[i]^2
        }
      }
    }
    gammaeta <- as.numeric(sapply(1:G,function(g)c(gamma[g,],eta[g])))
    #    Ut <- Ut - 2*pen*gammaeta
    #    It <- It - 2*pen*diag(length(gammaeta))
    if (pZ > pZ.tilde) {
      Ut[(pZ.tilde+1):pZ] <- 0
      It[(pZ.tilde+1):pZ,(pZ.tilde+1):pZ] <- diag(pZ-pZ.tilde)
      old.gammaeta <- gammaeta
      gammaeta <- old.gammaeta - solve(It,Ut)
      if (any(abs(gammaeta) > bound)) {
        step <- - solve(It,Ut)
        scale <- min(abs(c(((bound - old.gammaeta) / step)[step > 0], ((old.gammaeta + bound) / step)[step < 0])))
        scale <- min(scale,1)
        gammaeta <- old.gammaeta + scale * step
        print("gamma bound enforced")
      }
      gammaeta[(pZ.tilde+1):pZ] <- 0
    } else {
      old.gammaeta <- gammaeta
      gammaeta <- old.gammaeta - solve(It,Ut)
      if (any(abs(gammaeta) > bound)) {
        step <- - solve(It,Ut)
        scale <- min(abs(c(((bound - old.gammaeta) / step)[step > 0], ((old.gammaeta + bound) / step)[step < 0])))
        scale <- min(scale,1)
        gammaeta <- old.gammaeta + scale * step
        print("gamma bound enforced")
      }
    }
    for (g in 1:G) {
      gamma[g,] <- gammaeta[((g-1)*(pZ+1)+1):((g-1)*(pZ+1)+pZ)]
      eta[g] <- gammaeta[g*(pZ+1)]
    }
    
    Eexpeb <- array(dim=c(n,G))
    for (i in 1:n) {
      for (g in 1:G) {
        Eexpeb[i,g] <- sum(exp(gridb[i,g,]*eta[g]) * weightb[i,g,])
      }
    }
    S0 <- rep(0,n)
    for (g in 1:G) {
      ebZg <- exp(baseZ%*%gamma[g,ph])
      S0g <- rep(0,n)
      sumS0 <- sum(fg[,g] * ebZg * Eexpeb[,g])
      S0g[1] <- sumS0
      accum0 <- 0
      for (i in 2:n) {
        accum0 <- accum0 + fg[i-1,g] * ebZg[i-1] * Eexpeb[i-1,g]
        if (D[i]==1 & !is.tie[i]) {
          S0g[i] <- S0g[i-1] - accum0
          accum0 <- 0
        } else S0g[i] <- S0g[i-1]
      }
      eBg <- exp(Bmat%*%gamma[g,pt])
      j <- 0
      for (i in which(D==1)) {
        if (!is.tie[i]) j <- j + 1
        S0g[i] <- S0g[i] * eBg[j]
      }
      S0 <- S0 + S0g
    }
    j <- 1
    for (i in 1:n) {
      if (D[i]==1 & !is.tie[i]) {
        haz[j] <- 1/S0[i]
        if (i == n) break
        for (k in (i+1):n) {
          if (is.tie[k]) haz[j] <- haz[j] + 1/S0[k] else break
        }
        j <- j + 1
      }
    }
    param <- rep(NA,length=length(c(alpha,beta,sigma2,xi,gamma,eta,haz)))
    cnt <- 1
    for (g in 1:G) {
      for (j in 1:(pW+1)) {
        param[cnt] <- alpha[g,j]
        cnt <- cnt + 1
      }
    }
    for (g in 1:G) {
      for (k in 1:K) {
        for (j in 1:(pX+1)) {
          param[cnt] <- beta[g,k,j]
          cnt <- cnt + 1
        }
      }
    }
    if (covar=="ind") {
      for (g in 1:G) {
        for (k in 1:K) {
          param[cnt] <- sigma2[g,k]
          cnt <- cnt + 1
        }
      }
    } else {
      for (g in 1:G) {
        for (k in 1:K) {
          for (q in 1:ns) {
            param[cnt] <- sigma2[g,k,q]
            cnt <- cnt + 1
          }
        }
      }
    }
    for (g in 1:G) {
      param[cnt] <- xi[g]
      cnt <- cnt + 1
    }
    for (g in 1:G) {
      for (j in 1:pZ) {
        param[cnt] <- gamma[g,j]
        cnt <- cnt + 1
      }
    }
    for (g in 1:G) {
      param[cnt] <- eta[g]
      cnt <- cnt + 1
    }
    for (j in 1:length(haz)) {
      param[cnt] <- haz[j]
      cnt <- cnt + 1
    }
    EMRes <- list(param,fg,gridb,weightb)
    names(EMRes) <- c("param","fg","gridb","weightb")
    return(EMRes)
  }
  
  set_param <- function(alpha,beta,sigma2,xi,gamma,eta,haz) {
    param <- rep(NA,length=length(c(alpha,beta,sigma2,xi,gamma,eta,haz)))
    cnt <- 1
    for (g in 1:G) {
      for (j in 1:(pW+1)) {
        param[cnt] <- alpha[g,j]
        cnt <- cnt + 1
      }
    }
    for (g in 1:G) {
      for (k in 1:K) {
        for (j in 1:(pX+1)) {
          param[cnt] <- beta[g,k,j]
          cnt <- cnt + 1
        }
      }
    }
    if (covar=="ind") {
      for (g in 1:G) {
        for (k in 1:K) {
          param[cnt] <- sigma2[g,k]
          cnt <- cnt + 1
        }
      }
    } else {
      for (g in 1:G) {
        for (k in 1:K) {
          for (q in 1:ns) {
            param[cnt] <- sigma2[g,k,q]
            cnt <- cnt + 1
          }
        }
      }
    }
    for (g in 1:G) {
      param[cnt] <- xi[g]
      cnt <- cnt + 1
    }
    for (g in 1:G) {
      for (j in 1:pZ) {
        param[cnt] <- gamma[g,j]
        cnt <- cnt + 1
      }
    }
    for (g in 1:G) {
      param[cnt] <- eta[g]
      cnt <- cnt + 1
    }
    for (j in 1:length(haz)) {
      param[cnt] <- haz[j]
      cnt <- cnt + 1
    }
    #param <- list(c(param))
    return(param)
  }
  
  update_param <- function(param) {
    cnt <- 1
    for (g in 1:G) {
      for (j in 1:(pW+1)) {
        alpha[g,j] <<- param[cnt]
        cnt <- cnt + 1
      }
    }
    for (g in 1:G) {
      for (k in 1:K) {
        for (j in 1:(pX+1)) {
          beta[g,k,j] <<- param[cnt]
          cnt <- cnt + 1
        }
      }
    }
    if (covar=="ind") {
      for (g in 1:G) {
        for (k in 1:K) {
          sigma2[g,k] <<- param[cnt]
          cnt <- cnt + 1
        }
      }
    } else {
      for (g in 1:G) {
        for (k in 1:K) {
          for (q in 1:ns) {
            sigma2[g,k,q] <<- param[cnt]
            cnt <- cnt + 1
          }
        }
      }
    }
    for (g in 1:G) {
      xi[g] <<- param[cnt]
      cnt <- cnt + 1
    }
    for (g in 1:G) {
      for (j in 1:pZ) {
        gamma[g,j] <<- param[cnt]
        cnt <- cnt + 1
      }
    }
    for (g in 1:G) {
      eta[g] <<- param[cnt]
      cnt <- cnt + 1
    }
    for (j in 1:length(haz)) {
      haz[j] <<- param[cnt]
      cnt <- cnt + 1
    }
    return(list(alpha=alpha,beta=beta,sigma2=sigma2,xi=xi,gamma=gamma,eta=eta,haz=haz))
  } 
  
  n <- length(Time)
  pW <- ncol(W)-1
  K <- dim(X)[2]
  ni <- matrix(ni,n,K)
  pX <- dim(X)[4]-1
  pZ.tilde <- dim(Z)[2]
  baseZ <- Z
  maxni <- dim(Y)[3]
  ind.mat <- rep(list(1),K)
  fg <- array(dim=c(n,G))
  if(!any(is.na(knots))){
    nknots <- length(knots)
  }
  for (k in 1:K) {
    ind.mat[[k]] <- t(sapply(1:n,function(i){
      a <- rep(NA,maxni)
      if (ni[i,k] > 0) a[1:ni[i,k]] <- 1
      a
    }))
  }
  ns <- as.numeric(covar=="exchange")+1
  #order vector
  ord <- order(Time)
  Y <- Y[ord,,,drop=FALSE]
  baseZ <- baseZ[ord,,drop=FALSE]
  X <- X[ord,,,,drop=FALSE]
  W <- W[ord,,drop=FALSE]
  ni <- ni[ord,,drop=FALSE]
  Time <- Time[ord]
  D <- D[ord]
  if (D[1]==0) {
    rm <- 1:(min(which(D==1))-1)
    n <- n - length(rm)
    Y <- Y[-rm,,,drop=FALSE]
    X <- X[-rm,,,,drop=FALSE]
    W <- W[-rm,,drop=FALSE]
    ni <- ni[-rm]
    Time <- Time[-rm]
    D <- D[-rm]
    baseZ <- baseZ[-rm,]
  }
  for (i in 1:n) {
    set <- setdiff(1:maxni,min(ni[i],1):ni[i])
    Y[i,,set] <- NA
    X[i,,set,] <- NA
  }
  
  is.tie <- rep(FALSE,n)
  for (i in 2:n) if (Time[i] == Time[i-1] & D[i] == 1) is.tie[i] <- TRUE
  Tt <- unique(Time[D==1])
  if(any(is.na(knots))){
    if (nknots>=1) {
      knots <- quantile(Tt,prob=seq(0,1,length=nknots+2))[-c(1,nknots+2)]
      Bmat <- bSpline(Tt,degree=degree,knots=knots,intercept=TRUE)
    } else Bmat <- matrix(1,nrow=length(Tt),ncol=1)
  } else Bmat <- bSpline(Tt,degree=degree,knots=knots,intercept=TRUE)
  pZ <- pZ.tilde + ncol(Bmat)
  #  Z <- array(dim=c(n,pZ,sum(D==1)))
  #  for (i in 1:n) Z[i,,] <- t(cbind(matrix(baseZ[i,],nrow=length(Tt),ncol=pZ.tilde,byrow=TRUE),Bmat))
  
  #nodeweight <- read.table("GQ3.txt",sep=",")
  
  #nodes <- c(-3.436159,-2.532732,-1.756684,-1.036611,-0.3429013,0.3429013,1.036611,1.756684,2.532732,3.436159)
  #weights <- c(7.640433e-06,0.001343646,0.03387439,0.2401386,0.6108626,0.6108626,0.2401386,0.03387439,0.001343646,7.640433e-06)
  
  #gq <- list(nodes=nodes,weights=weights)
  #gq <- list(nodes=as.numeric(nodeweight[1,]),weights=as.numeric(nodeweight[2,]))
  gq <- gauss.quad(h,kind="hermite")
  
  upto <- match(Time,Tt)
  for (i in 1:n) if (is.na(upto[i])) upto[i] <- upto[i-1]
  if (length(init.param) == 0) {
    if (covar=="ind") sigma2 <- array(dim=c(G,K)) else sigma2 <- array(dim=c(G,K,2))
    alpha <- matrix(0,G,pW+1)
    beta <- array(dim=c(G,K,(pX+1)))
    xi <- rep(0.1,G)
    eta <- rep(0,G)
    gamma <- array(dim=c(G,(ncol(W)-1)))

    # longX1 <- sapply(1:dim(X)[4],function(s)as.numeric(X[,1,,s]))
    # longY1 <- as.numeric(Y[,1,])
    # longX2 <- sapply(1:dim(X)[4],function(s)as.numeric(X[,2,,s]))
    # longY2 <- as.numeric(Y[,2,])
    longX <- array(NA,dim=c(dim(X)[1]*dim(X)[3],dim(X)[4],K))
    longY <- array(NA,dim=c(dim(Y)[1]*dim(Y)[3],K))
    for(i in 1:K){
      longX[,,i] <- sapply(1:dim(X)[4],function(s)as.numeric(X[,i,,s]))
      longY[,i] <- as.numeric(Y[,i,])
    }
    set.seed(seed*123456)
    ytmp1 <- Y[,,1]
    ytmp1[is.na(ytmp1)] <- mean(ytmp1[!is.na(ytmp1)])
    clust <- kmeans(cbind(Time,D,ytmp1),G)$cluster
    longC <- rep(clust,times=dim(Y)[3])
    if (G > 1) fitW <- multinom(clust ~ 0+W)
    if (G > 1) {if (G > 2) for (j in 1:(G-1)) alpha[j,] <- rbind(0,summary(fitW)$coefficients)[j,] - summary(fitW)$coefficients[G-1,] else
      alpha[1,] <- -summary(fitW)$coefficients}
    for (c in unique(clust)){
      for(i in 1:K){
        fit <- lm(longY[longC==c,i]~0+longX[longC==c,,i])
        beta[c,i,] <- fit$coefficient
        if (covar=="ind"){
          sigma2[c,] <- summary(fit)$sigma^2/2
        }else if (covar=="exchange"){
          sigma2[c,i,] <- c(summary(fit)$sigma^2/2,summary(fit)$sigma^2/2)
        }else{
          print("Please input a correct value for \"covar\". ")
        }
      }
      gamma[c,] <- coxph(Surv(Time[clust==c],D[clust==c])~baseZ[clust==c,],ties="breslow")$coefficient
    }
    # for (c in unique(clust)) {
    #   fit <- lm(longY1[longC==c]~0+longX1[longC==c,])
    #   beta[c,1,] <- fit$coefficient
    #   if (covar=="ind"){
    #     sigma2[c,1] <- summary(fit)$sigma^2/2
    #   }else{
    #     sigma2[c,1,] <- c(summary(fit)$sigma^2/2,summary(fit)$sigma^2/2)
    #   }
    #   fit <- lm(longY2[longC==c]~0+longX2[longC==c,])
    #   beta[c,2,] <- fit$coefficient
    #   if (covar=="ind"){
    #     sigma2[c,2] <- summary(fit)$sigma^2/2
    #   }else{
    #     sigma2[c,2,] <- c(summary(fit)$sigma^2/2,summary(fit)$sigma^2/2)
    #   }
    #   gamma[c,] <- coxph(Surv(Time[clust==c],D[clust==c])~baseZ[clust==c,],ties="breslow")$coefficient
    # }
    gamma <- cbind(gamma,matrix(0,nrow=G,ncol=nknots+degree+1))
    # is.tie <- rep(FALSE,length(D))
    # for (i in 2:length(D)) if (Time[i] == Time[i-1]) is.tie[i] <- TRUE
    # haz <- rep(0,length=sum(D==1&!is.tie))
    # for (i in 1:length(Time)) {
    #   if (D[i] == 1 & !is.tie[i]) {
    #     partsum <- 0
    #     for (k in i:length(Time)) partsum <- partsum + exp(sum(baseZ[k,]*gamma[clust[k],1:2]))
    #     haz[j] <- 1/partsum
    #     j <- j + 1
    #   }
    # }
    # haz <- 1/length(Tt) 
    haz <- 0.5*(Tt-c(0,Tt[-length(Tt)]))

  } else {
    alpha <- init.param$alpha
    beta <- init.param$beta
    sigma2 <- init.param$sigma2
    xi <- init.param$xi
    eta <- init.param$eta
    gamma <- init.param$gamma
    haz <- init.param$haz
  }
  gridb <- array(dim=c(n,G,h))
  weightb <- array(dim=c(n,G,h))
  
  param <- set_param(alpha,beta,sigma2,xi,gamma,eta,haz)
  diff <- 10
  
  it <- 1
  fg <- 0
  start.like <- FALSE
  accel.on <- FALSE
  if (start.like) like <- loglike() else like <- -1e50
  iter <- 1
  while (iter < max.iter & (diff > epsilon2 | diff < 0)) {
    like.diff <- like.diff1
    old.param <- param
    old.like <- like
    EMRes <- EMStep(fg,weightb,gridb)
    param1 <- EMRes$param
    fg <- EMRes$fg
    gridb <- EMRes$gridb
    weightb <- EMRes$weightb
    updateparam <- update_param(param=param1)
    alpha <- updateparam$alpha
    beta <- updateparam$beta
    sigma2 <- updateparam$sigma2
    xi <- updateparam$xi
    gamma <- updateparam$gamma
    eta <- updateparam$eta
    haz <- updateparam$haz
    param2 <- param1
    
    if ((diff < epsilon & accelEM) | accel.on) {
      start.like <- TRUE
      accel.on <- TRUE
      like.diff <- like.diff2
      EMRes <- EMStep(fg,weightb,gridb)
      param2 <- EMRes$param
      fg <- EMRes$fg
      gridb <- EMRes$gridb
      weightb <- EMRes$weightb
      #update_param(param=param2,alpha=alpha,beta=beta,sigma2=sigma2,xi=xi,gamma=gamma,eta=eta,haz=haz)
      updateparam <- update_param(param=param2)
      alpha <- updateparam$alpha
      beta <- updateparam$beta
      sigma2 <- updateparam$sigma2
      xi <- updateparam$xi
      gamma <- updateparam$gamma
      eta <- updateparam$eta
      haz <- updateparam$haz
      r <- param1 - old.param
      v <- param2 - param1 - r
      a <- - sqrt(sum(r^2)) / sqrt(sum(v^2))
      newparam <- old.param - 2 * a * r + a^2 * v
      #update_param(param=newparam,alpha=alpha,beta=beta,sigma2=sigma2,xi=xi,gamma=gamma,eta=eta,haz=haz)
      updateparam <- update_param(param=newparam)
      alpha <- updateparam$alpha
      beta <- updateparam$beta
      sigma2 <- updateparam$sigma2
      xi <- updateparam$xi
      gamma <- updateparam$gamma
      eta <- updateparam$eta
      haz <- updateparam$haz
      if (any(abs(newparam) > bound) | any(haz < 0) | any(xi < 0) | any(sigma2 < 0)) like <- old.like - 1e50 else {
        EMRes <- EMStep(fg,weightb,gridb)
        param <- EMRes$param
        fg <- EMRes$fg
        gridb <- EMRes$gridb
        weightb <- EMRes$weightb
        #update_param(param=param,alpha=alpha,beta=beta,sigma2=sigma2,xi=xi,gamma=gamma,eta=eta,haz=haz)
        updateparam <- update_param(param=param)
        alpha <- updateparam$alpha
        beta <- updateparam$beta
        sigma2 <- updateparam$sigma2
        xi <- updateparam$xi
        gamma <- updateparam$gamma
        eta <- updateparam$eta
        haz <- updateparam$haz
      }
    }
    if (like < old.like | diff > epsilon) {
      param <- param2
      #update_param(param=param,alpha=alpha,beta=beta,sigma2=sigma2,xi=xi,gamma=gamma,eta=eta,haz=haz)
      updateparam <- update_param(param=param)
      alpha <- updateparam$alpha
      beta <- updateparam$beta
      sigma2 <- updateparam$sigma2
      xi <- updateparam$xi
      gamma <- updateparam$gamma
      eta <- updateparam$eta
      haz <- updateparam$haz
      if (start.like) like <- loglike()
    }
    if (like.diff) {like <- loglike();diff <- like - old.like} else diff <- sum(abs(old.param-param))
    if (start.like) cat(c(ifelse(start.like,"like_diff ","param_diff "),diff,"; log-likelihood ",like,"\n")) else
      cat(c(ifelse(start.like,"like_diff ","param_diff "),diff,"\n"))
    it <- it + 1
  }
  if (cal.inf) Information <- calInf() else Information <- NULL
  Haz <- array(NA,dim=c(length(haz),G))
  for(g in 1:G){
    Haz[,g] <- sapply(1:length(haz),function(i){
      Haz <- haz[i] * exp(sum(gamma[g,(pZ.tilde+1):pZ]*Bmat[i,]))
      Haz
    })
  }
  list(alpha=alpha,beta=beta,sigma2=sigma2,xi=xi,gamma=gamma,eta=eta,Tt=Tt,Haz=Haz,Bmat=Bmat,post.prob=fg,gridb=gridb,weightb=weightb,Information=Information,loglike=like)
}



#' Generate Dataset
#' 
#' @description Create a dataset based on the setting of the simulation studies in Wong et al. (2022)
#'
#' @param n Sample size
#' @param seed Seed of the random generator (optional)
#' @return A list of the following components: \itemize{  
#'   \item \strong{Y} :  An \eqn{(n \times J \times m)} array of longitudinal outcome measurements, where \eqn{n} is the sample size, \eqn{J} is the number of longitudinal measurement types, and
#' \eqn{m} is the maximum number of measurement times. It can contain \code{NA} values if the number of measurements for a subject is fewer than the maximum number of measurements. The \eqn{(i,j,k)}th element corresponds to the \eqn{k}th measurement of the \eqn{j}th type of longitudinal outcome for the \eqn{i}th subject
#'   \item \strong{X} : An \eqn{(n \times J \times m \times p_X)} array of covariates (excluding intercept) of the longitudinal outcome model, 
#' where \eqn{n} is the sample size, \eqn{J} is the number of longitudinal measurement types, \eqn{m} is the number of measurement times, and \eqn{p_X} is the number of covariates. 
#' The \eqn{(i,j,k,l)}th element corresponds to the \eqn{l}th covariate for the \eqn{k}th measurement of the \eqn{j}th type of longitudinal outcome for the \eqn{i}th subject
#'   \item \strong{W} : An \eqn{(n \times p_W)} matrix of covariates for the latent class regression model, where \eqn{n} is the sample size, and 
#' \eqn{p_W} is the number of covariate. The \eqn{(i,l)}th element corresponds to the \eqn{l}th covariate for the \eqn{i}th subject
#'   \item \strong{Time} : An \eqn{n}-vector of observed event or censoring times
#'   \item \strong{D} : An \eqn{n}-vector of event indicators
#'   \item \strong{ni} : An \eqn{(n \times J)} matrix of numbers of measurements for the longitudinal outcomes
#'   \item \strong{Z} : An \eqn{(n \times p_Z)} matrix of time-independent covariates for the survival model, where \eqn{n} is the sample size, and 
#' \eqn{p_Z} is the number of covariates. The \eqn{(i,l)}th element corresponds to the \eqn{l}th covariate for the \eqn{i}th subject
#'   }
#'   Based on the setting of the simulation studies in Wong et al. (2022), we fix \eqn{J = 2}, \eqn{m = 10}, \eqn{p_X = 3}, \eqn{p_W = 2}, and \eqn{p_Z = 2}.
#' @examples dataset <- create.data(n=1000)
#' @references Wong, K. Y., Zeng, D., & Lin, D. Y. (2022). Semiparametric latent-class models for multivariate longitudinal and survival data. The Annals of Statistics. 50 487–510. 
#' @export


create.data <- function(n, seed=1){
  
  K <- 2
  pW <- 2
  pX <- 3
  maxni <- 10
  G <- 3
  alpha0 <- matrix(0,nrow=G,ncol=pW+1)
  alpha0[1,] <- c(-0.25,0.5,-0.5)
  alpha0[2,] <- c(-0.25,0.5,0.5)
  beta0 <- array(0,dim=c(G,K,pX+1))
  tmp <- expand.grid(rep(list(c(-1,1)),pX+1))
  tmp <- tmp[c(1,16,7,10,8,5,12,4,13,3,14),]
  tmp[3,1] <- 0
  tmp[4,1] <- 0
  it <- 1
  for (g in 1:G) {
    for (k in 1:K) {
      beta0[g,k,] <- as.numeric(tmp[it,])
      it <- it + 1
    }
  }
  beta0[,1,][3,2] <- 0
  beta0[,1,][2,3] <- 0
  beta0[,1,][2,4] <- 1
  beta0[,2,][3,2] <- 0
  beta0[,2,][1,3] <- 0
  beta0[,2,][1,4] <- 0
  eta0 <- -0.2*(1:G)+0.3
  if (G==2) lambda0g <- c(0.5,0.25)
  if (G==3) lambda0g <- c(0.5,0.25,1)
  gamma0g <- matrix(c(0.1*(1:G),0.1*(G:1))-0.2,G,2)
  sigma20 <- seq(0.5,1.5,length=G)
  set.seed(seed*1357+2468)
  W <- matrix(nrow=n,ncol=pW+1)
  eWa <- matrix(nrow=n,ncol=G)
  probG0 <- matrix(nrow=n,ncol=G)
  C <- rep(NA,n)
  Y <- array(dim=c(n,K,maxni))
  X <- array(dim=c(n,K,maxni,pX+1))
  ni <- rep(NA,n)
  baseX <- array(dim=c(n,pX-1))
  T <- rep(0,n)
  D <- rep(NA,n)
  trueT <- rep(0,n)
  for (i in 1:n) {
    time.seq <- seq(0,by=0.15,length=maxni) 
    spec.b <- rnorm(K)
    X1 <- rbinom(1,1,0.5)
    X2 <- rnorm(1)
    W[i,] <- c(1,X1,X2)
    for (g in 1:G) eWa[i,g] <- exp(W[i,]%*%as.vector(alpha0[g,]))
    probG0[i,] <- eWa[i,] / sum(eWa[i,])
    C[i] <- which(rmultinom(1, 1, probG0[i,])==1)
    
    b <- rnorm(1)*sqrt(c(0.5,1,1.5)[C[i]])
    baseX[i,] <- c(X1,X2)
    for (k in 1:K) {
      for (j in 1:maxni) {
        X[i,k,j,] <- c(1,time.seq[j],X1,X2)
        Y[i,k,j] <- sum(X[i,k,j,]*beta0[C[i],k,])+b+spec.b[k]+rnorm(1)*sqrt(sigma20[C[i]])
      }
    }
    Z <- c(X1,X2)
    u <- runif(1)
    if (C[i] == 2) t <- log(1 - lambda0g[C[i]] * log(u) / exp(eta0[C[i]]*b+sum(Z*gamma0g[C[i],]))) / lambda0g[C[i]] else t <- -log(u) / (exp(eta0[C[i]]*b+sum(Z*gamma0g[C[i],])) * lambda0g[C[i]])
    c <- runif(1,0,5)
    trueT[i] <- t
    T[i] <- min(c,t)
    D[i] <- as.numeric(t<c)
    ni[i] <- sum(time.seq < T[i])
  }
  dataset <- list(Y,X,W,T,D,ni,baseX)
  names(dataset) <- c("Y","X","W","Time","D","ni","Z")
  return(dataset)
}


