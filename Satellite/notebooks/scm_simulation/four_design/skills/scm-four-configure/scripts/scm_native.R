# The design choices Synth lacks, as a copy of Synth 1.1-10's synth.R and fn.V.R (CRAN source, code/Synth/R/) with
# a branch at each choice. With weights = "convex" and interpolation_penalty = "none" every line Synth runs is run,
# in Synth's order, so the result is Synth's.
#
#   weights = "extrapolation"        (ADH 2010 p. 496: "negative weights or weights larger than one can be used at
#                                    the cost of allowing extrapolation") the weights sum to one and have no bounds.
#                                    The V-weighted least squares under that one constraint is solved through its
#                                    linear optimality equations; MASS::ginv gives the minimum norm solution when the
#                                    solution is not unique.
#   interpolation_penalty = "distance" (p. 496: "penalty terms specified as increasing functions of the distances
#                                    between X1 and the corresponding values for the control units with positive
#                                    weights in W") adds penalty_lambda * sum_j w_j * d_j, d_j the V-weighted squared
#                                    distance between the treated unit's scaled predictors and donor j's. In ipop's
#                                    form (c'x + x'Hx/2, half of Synth's loss) this adds penalty_lambda * d / 2 to c.
#                                    Allowed with convex weights only.
# Changed lines are marked "# native". Uses scm_data, scm_dataprep, scm_custom_v and scm_record from synth_fit.R.
suppressPackageStartupMessages({library(Synth); library(kernlab); library(optimx); library(MASS)})

solve.w.native <- function(X0.scaled, X1.scaled, V, quadopt, margin.ipop, sigf.ipop, bound.ipop,   # native
                           weights, penalty.lambda) {                                               # native
    if (weights == "extrapolation") {                                                               # native
      H <- t(X0.scaled) %*% V %*% (X0.scaled)                                                       # native
      K <- rbind(cbind(H, rep(1, ncol(H))), c(rep(1, ncol(H)), 0))                                  # native
      rhs <- c(c(t(X0.scaled) %*% V %*% X1.scaled), 1)                                              # native
      return(as.matrix(as.numeric(ginv(K) %*% rhs)[1:ncol(H)]))                                     # native
    }                                                                                               # native
    # set up QP problem
    H <- t(X0.scaled) %*% V %*% (X0.scaled)
    a <- X1.scaled
    c <- -1*c(t(a) %*% V %*% (X0.scaled) )
    if (!is.null(penalty.lambda)) {                                                                 # native
      dist <- apply(X0.scaled, 2, function(x0) as.numeric(t(a - x0) %*% V %*% (a - x0)))            # native
      c <- c + penalty.lambda * dist / 2                                                            # native
    }                                                                                               # native
    A <- t(rep(1, length(c)))
    b <- 1
    l <- rep(0, length(c))
    u <- rep(1, length(c))
    r <- 0

    if (quadopt == "ipop") {
      res <- ipop(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                  margin = margin.ipop, maxiter = 1000,
                  sigf = sigf.ipop, bound = bound.ipop)
      solution.w <- as.matrix(primal(res))
    } else if (quadopt == "LowRankQP") {
      stop("LowRankQP is no longer supported; please use quadopt = \"ipop\" instead")
    } else {
      stop(sprintf("Unknown quadopt = \"%s\"; supported value is \"ipop\"", quadopt))
    }
    solution.w
}

fn.V.native <-
function(
           variables.v = stop("variables.v missing"),
           X0.scaled = stop("X0.scaled missing"),
           X1.scaled = stop("X1.scaled missing"),
           Z0 = stop("Z0 missing"),
           Z1 = stop("Z1 missing"),
           margin.ipop = 0.0005,
           sigf.ipop = 5,
           bound.ipop = 10,
           quadopt = "ipop",
           weights = "convex",                                                                      # native
           penalty.lambda = NULL                                                                    # native
           )

  {

    # check quadopt
    if (!(quadopt %in% c("ipop", "LowRankQP"))) {
      stop("option quadopt must be \"ipop\" (\"LowRankQP\" is no longer supported)")
    }

    # rescale par
    #V <- diag( abs(variables.v)/sum(abs(variables.v)) )
    V <- diag(x=as.numeric(abs(variables.v)/sum(abs(variables.v))),
              nrow=length(variables.v),ncol=length(variables.v))

    solution.w <- solve.w.native(X0.scaled, X1.scaled, V, quadopt, margin.ipop, sigf.ipop,          # native
                                 bound.ipop, weights, penalty.lambda)                               # native

    # compute losses
    loss.w <- as.numeric(t(X1.scaled - X0.scaled %*% solution.w) %*%
      (V) %*% (X1.scaled - X0.scaled %*% solution.w))

    loss.v <- as.numeric(t(Z1 - Z0 %*% solution.w) %*%
      ( Z1 - Z0 %*% solution.w ))
    loss.v <- loss.v/nrow(Z0)

    return(invisible(loss.v))
  }

synth.native <-
function(           data.prep.obj = NULL,
                      X1 = NULL,
                      X0 = NULL,
                      Z0 = NULL,
                      Z1 = NULL,
                      custom.v = NULL,
                      optimxmethod = c("Nelder-Mead","BFGS"),
                      genoud = FALSE,
                      quadopt = "ipop",
                      Margin.ipop = 0.0005,
                      Sigf.ipop = 5,
                      Bound.ipop = 10,
                      verbose = FALSE,
                      weights = "convex",                                                           # native
                      penalty.lambda = NULL,                                                        # native
                       ...
                      )
  {
    # Retrieve dataprep objects
    if (!is.null(data.prep.obj)) {
      if (verbose) cat("\nX1, X0, Z1, Z0 all come directly from dataprep object.\n\n")
      X1 <- data.prep.obj$X1
      Z1 <- data.prep.obj$Z1
      X0 <- data.prep.obj$X0
      Z0 <- data.prep.obj$Z0
    } else {
      if (verbose) cat("X1, X0, Z1, Z0 were individually input (not dataprep object).\n\n")
    }

     # routine checks
     store <- list(X1=X1,X0=X0,Z1=Z1,Z0=Z0)
     for(i in 1:4){
     if(is.null(store[[i]]))
      {stop(paste("\n",names(store)[i],"is missing \n"))}
     if(sum(is.na(store[[i]]))>0)
      {stop(paste("\n NAs in",names(store)[i],"\n"))}
     if(is.matrix(store[[i]]) == FALSE)
      {stop(paste("\n",names(store)[i],"is not a matrix object\n"))}
     }

    # geometry checks
    if(ncol(X1)!=1){stop("\n Please specify only one treated unit: X1 has to have ncol= 1")}
    if(ncol(Z1)!=1){stop("\n Please specify only one treated unit: Z1 has to have ncol= 1")}

    if(ncol(X0)<2){stop("\n Please specify at least two control units: X0 has to have ncol >= 2 ")}
    if(ncol(Z0)<2){stop("\n Please specify at least two control units: Z0 has to have ncol >= 2")}

    if(nrow(Z0)!=nrow(Z1)){stop("\n Different number of periods for treated and controls: nrow(Z0) unequal nrow(Z1)")}
    if(nrow(X0)!=nrow(X1)){stop("\n Different number of predictors for treated and controls: nrow(X0) unequal nrow(X1)")}

    if(nrow(X0)==0){stop("No predictors specified. Please specify at least one predictor")}
    if(nrow(Z0)==0){stop("No periods specified for Z1 and Z0. Please specify at least one period")}

    if(0 %in% apply(X0,1,sd))
     {stop("\n At least one predictor in X0 has no variation across control units. Please remove this predictor.")}

    # Normalize X
    nvarsV <- dim(X0)[1]
    big.dataframe <- cbind(X0, X1)
    divisor <- sqrt(apply(big.dataframe, 1, var))
    scaled.matrix <-
      t(t(big.dataframe) %*% ( 1/(divisor) *
                              diag(rep(dim(big.dataframe)[1], 1)) ))

    X0.scaled <- scaled.matrix[, 1:ncol(X0), drop = FALSE]
    X1.scaled <- scaled.matrix[, ncol(scaled.matrix)]


    # check if custom v weights are supplied or
    # if only on predictor is specified,
    # we jump to quadratic optimization over W weights
    # if not start optimization over V and W
    if(is.null(custom.v) & nrow(X0) != 1)
      {

      # two attempts for best V are made:
      # equal weights and regression based starting values
      if (verbose) cat("\n****************",
                       "\n searching for synthetic control unit  \n", "\n")

      if (genoud == TRUE) { # if user wants genoud as well
      # we run genoud first
      if (verbose) cat("\n****************",
                       "\n genoud() requested for optimization\n", "\n")

      rgV.genoud <- rgenoud::genoud(
                             fn.V.native,                                                           # native
                             nvarsV,
                             X0.scaled = X0.scaled,
                             X1.scaled = X1.scaled,
                             Z0 = Z0,
                             Z1 = Z1,
                             quadopt = quadopt,
                             margin.ipop = Margin.ipop,
                             sigf.ipop = Sigf.ipop,
                             bound.ipop = Bound.ipop,
                             weights = weights,                                                     # native
                             penalty.lambda = penalty.lambda                                        # native
                             )
      SV1 <- rgV.genoud$par  # and use these as starting values

      if (verbose) cat("\n****************",
                       "\n genoud() finished, now running local optimization using optim()\n", "\n")

      } else {
      # if we don't use genoud first: set of starting values: equal weights
      SV1 <- rep(1/nvarsV,nvarsV)
      }

      # now we run optimization
      all.methods <- FALSE
      if(sum(optimxmethod %in% c("All"))==1){ all.methods <- TRUE }
     rgV.optim.1 <- optimx(par=SV1, fn=fn.V.native,                                                # native
                             gr=NULL, hess=NULL,
                             method=optimxmethod, itnmax=NULL, hessian=FALSE,
                             control=list(kkt=FALSE,
                                          starttests=FALSE,
                                          dowarn=FALSE,
                                          all.methods=all.methods),
                             X0.scaled = X0.scaled,
                             X1.scaled = X1.scaled,
                             Z0 = Z0,
                             Z1 = Z1,
                             quadopt = quadopt,
                             margin.ipop = Margin.ipop,
                             sigf.ipop = Sigf.ipop,
                             bound.ipop = Bound.ipop,
                             weights = weights,                                                     # native
                             penalty.lambda = penalty.lambda                                        # native
                            )
      # get minimum
      if(verbose==TRUE){print(rgV.optim.1)}
      rgV.optim.1 <- collect.optimx(rgV.optim.1,"min")

      # second set of starting values: regression method
      # will sometimes not work because of collinear Xs
      # so it's wrapped in a try command
      Xall <- cbind(X1.scaled,X0.scaled)
      Xall <- cbind(rep(1,ncol(Xall)),t(Xall))
      Zall <- cbind(Z1,Z0)
      Beta <- try(solve(t(Xall)%*%Xall)%*%t(Xall)%*%t(Zall),silent=TRUE)

      # if inverses did not work, we
      # stick with first results
      if(inherits(Beta,"try-error"))
       {
        rgV.optim <- rgV.optim.1
       } else {
      # otherwise we run a second optimization with regression starting values
        Beta <- Beta[-1,]
        V    <- Beta%*%t(Beta)
        SV2  <- diag(V)
        SV2 <- SV2 / sum(SV2)

      rgV.optim.2 <- optimx(par=SV2, fn=fn.V.native,                                                # native
                             gr=NULL, hess=NULL,
                             method=optimxmethod, itnmax=NULL, hessian=FALSE,
                             control=list(kkt=FALSE,
                                          starttests=FALSE,
                                          dowarn=FALSE,
                                          all.methods=all.methods),
                             X0.scaled = X0.scaled,
                             X1.scaled = X1.scaled,
                             Z0 = Z0,
                             Z1 = Z1,
                             quadopt = quadopt,
                             margin.ipop = Margin.ipop,
                             sigf.ipop = Sigf.ipop,
                             bound.ipop = Bound.ipop,
                             weights = weights,                                                     # native
                             penalty.lambda = penalty.lambda                                        # native
                            )
      # get minimum
      if(verbose==TRUE){print(rgV.optim.2)}
      rgV.optim.2 <- collect.optimx(rgV.optim.2,"min")

      # ouput
      if(verbose == TRUE){
      cat("\n Equal weight loss is:",rgV.optim.1$value,"\n")
      cat("\n Regression Loss is:",rgV.optim.2$value,"\n")
      }
      # and keep the better optim results
      if(rgV.optim.1$value < rgV.optim.2$value)
       {
        rgV.optim <- rgV.optim.1
       } else {
        rgV.optim <- rgV.optim.2
       }
      } # close if statement for second regression based optimization attempt

      # final V weights from optimization
      solution.v   <- abs(rgV.optim$par)/sum(abs(rgV.optim$par))
     } else { # jump here if only optimize over W

     if (verbose) cat("\n****************",
                      "\n optimization over w weights: computing synthetic control unit \n", "\n\n")

     if (nrow(X0) == 1) {
       custom.v <- 1 # only one predictor: V is the identity matrix
     } else {
       # user-supplied v
       if (verbose) cat("\n****************",
                        "\n v weights supplied manually: computing synthetic control unit \n", "\n\n")
       if (length(custom.v) != nvarsV) {
         stop("custom.v misspecified: length(custom.v) != nrow(X0)")
       }
       if (mode(custom.v) != "numeric") {
         stop("custom.v must be numeric")
       }
     }

    # enter solution.V
    rgV.optim  <- NULL
    solution.v <- abs(custom.v)/sum(custom.v)

  } # close else statment for by-passing V optimization


    # last step: now recover solution.w
    V <- diag(x=as.numeric(solution.v),nrow=nvarsV,ncol=nvarsV)
    solution.w <- solve.w.native(X0.scaled, X1.scaled, V, quadopt, Margin.ipop, Sigf.ipop,          # native
                                 Bound.ipop, weights, penalty.lambda)                               # native

    rownames(solution.w) <- colnames(X0)
    colnames(solution.w) <- "w.weight"
    names(solution.v) <- rownames(X0)

    loss.w <- t(X1.scaled - X0.scaled %*% solution.w) %*%
      V %*% (X1.scaled - X0.scaled %*% solution.w)

    loss.v <-
      t(Z1 - Z0 %*% as.matrix(solution.w)) %*%
        (Z1 - Z0 %*% as.matrix(solution.w))
    loss.v <- loss.v/nrow(Z0)

    # produce viewable output
    if (verbose) {
      cat("\n****************",
          "\n****************",
          "\n****************",
          "\n\nMSPE (LOSS V):", loss.v,
          "\n\nsolution.v:\n", round(as.numeric(solution.v), 10),
          "\n\nsolution.w:\n", round(as.numeric(solution.w), 10),
          "\n\n")
    }

    optimize.out <- list(
                         solution.v = solution.v,
                         solution.w = solution.w,
                         loss.v = loss.v,
                         loss.w = loss.w,
                         custom.v = custom.v,
                         rgV.optim = rgV.optim
                         )

    return(invisible(optimize.out))

  }

fit_native <- function(d, s) {
  dp <- scm_dataprep(d, s)
  so <- synth.native(data.prep.obj = dp, custom.v = scm_custom_v(s), optimxmethod = unlist(s$optimxmethod),
                     genoud = s$genoud, quadopt = s$quadopt, Margin.ipop = s$margin_ipop, Sigf.ipop = s$sigf_ipop,
                     Bound.ipop = s$bound_ipop, weights = s$weights,
                     penalty.lambda = if (s$interpolation_penalty == "distance") s$penalty_lambda else NULL)
  scm_record(d, dp, so, s, "native")
}
