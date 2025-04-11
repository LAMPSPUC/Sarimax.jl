class OCSBTest(_SeasonalStationarityTest):
    """Perform an OCSB test of seasonality.

    Compute the Osborn, Chui, Smith, and Birchenhall (OCSB) test for an input
    time series to determine whether it needs seasonal differencing. The
    regression equation may include lags of the dependent variable. When
    ``lag_method`` = "fixed", the lag order is fixed to ``max_lag``; otherwise,
    ``max_lag`` is the maximum number of lags considered in a lag selection
    procedure that minimizes the ``lag_method`` criterion, which can be
    "aic", "bic" or corrected AIC, "aicc".

    Critical values for the test are based on simulations, which have been
    smoothed over to produce critical values for all seasonal periods

    Parameters
    ----------
    m : int
        The seasonal differencing term. For monthly data, e.g., this would be
        12. For quarterly, 4, etc. For the OCSB test to work, ``m`` must
        exceed 1.

    lag_method : str, optional (default="aic")
        The lag method to use. One of ("fixed", "aic", "bic", "aicc"). The
        metric for assessing model performance after fitting a linear model.

    max_lag : int, optional (default=3)
        The maximum lag order to be considered by ``lag_method``.

    References
    ----------
    .. [1] Osborn DR, Chui APL, Smith J, and Birchenhall CR (1988)
           "Seasonality and the order of integration for consumption",
           Oxford Bulletin of Economics and Statistics 50(4):361-377.

    .. [2] R's forecast::OCSB test source code: https://bit.ly/2QYQHno
    """

    _ic_method_map = {
        "aic": lambda fit: fit.aic,
        "bic": lambda fit: fit.bic,
        # TODO: confirm False for add_constant, since the model fit contains
        #   . a constant term
        "aicc": lambda fit: _aicc(fit, fit.nobs, False),
    }

    def __init__(self, m, lag_method="aic", max_lag=3):
        super(OCSBTest, self).__init__(m=m)

        self.lag_method = lag_method
        self.max_lag = max_lag

    @staticmethod
    def _calc_ocsb_crit_val(m):
        """Compute the OCSB critical value"""
        # See:
        # https://github.com/robjhyndman/forecast/blob/
        # 8c6b63b1274b064c84d7514838b26dd0acb98aee/R/unitRoot.R#L409
        log_m = np.log(m)
        return (
            -0.2937411 * np.exp(
                -0.2850853 * (log_m - 0.7656451) + (-0.05983644) * (
                    (log_m - 0.7656451) ** 2
                )
            ) - 1.652202
        )

    @staticmethod
    def _do_lag(y, lag, omit_na=True):
        """Perform the TS lagging"""
        n = y.shape[0]
        if lag == 1:
            return y.reshape(n, 1)

        # Create a 2d array of dims (n + (lag - 1), lag). This looks cryptic..
        # If there are tons of lags, this may not be super efficient...
        out = np.ones((n + (lag - 1), lag)) * np.nan
        for i in range(lag):
            out[i: i + n, i] = y

        if omit_na:
            out = out[~np.isnan(out).any(axis=1)]
        return out

    @staticmethod
    def _gen_lags(y, max_lag, omit_na=True):
        """Create the lagged exogenous array used to fit the linear model"""
        if max_lag <= 0:
            return np.zeros(y.shape[0])

        # delegate down
        return OCSBTest._do_lag(y, max_lag, omit_na)

    @staticmethod
    def _fit_ocsb(x, m, lag, max_lag):
        """Fit the linear model used to compute the test statistic"""
        y_first_order_diff = diff(x, m)

        # if there are no more samples, we have to bail
        if y_first_order_diff.shape[0] == 0:
            raise ValueError(
                "There are no more samples after a first-order "
                "seasonal differencing. See http://alkaline-ml.com/pmdarima/"
                "seasonal-differencing-issues.html for a more in-depth "
                "explanation and potential work-arounds."
            )

        y = diff(y_first_order_diff)
        ylag = OCSBTest._gen_lags(y, lag)

        if max_lag > -1:
            # y = tail(y, -maxlag)
            y = y[max_lag:]

        # A constant term is added in the R code's lm formula. We do that in
        # the linear model's constructor
        mf = ylag[: y.shape[0]]
        ar_fit = sm.OLS(y, add_constant(mf)).fit(method="qr")

        # Create Z4
        z4_y = y_first_order_diff[lag:]  # new endog
        z4_lag = OCSBTest._gen_lags(y_first_order_diff, lag)[
            : z4_y.shape[0], :
        ]
        z4_preds = ar_fit.predict(add_constant(z4_lag))  # preds
        z4 = z4_y - z4_preds  # test residuals

        # Create Z5. Looks odd because y and lag depend on each other and go
        # back and forth for two stages
        z5_y = diff(x)
        z5_lag = OCSBTest._gen_lags(z5_y, lag)
        z5_y = z5_y[lag:]
        z5_lag = z5_lag[: z5_y.shape[0], :]
        z5_preds = ar_fit.predict(add_constant(z5_lag))
        z5 = z5_y - z5_preds

        # Finally, fit a linear regression on mf with z4 & z5 features added
        data = np.hstack(
            (
                mf,
                z4[: mf.shape[0]].reshape(-1, 1),
                z5[: mf.shape[0]].reshape(-1, 1),
            )
        )

        return sm.OLS(y, data).fit(method="qr")

    def _compute_test_statistic(self, x):
        m = self.m
        maxlag = self.max_lag
        method = self.lag_method

        # We might try multiple lags in this case
        crit_regression = None
        if maxlag > 0 and method != "fixed":
            try:
                icfunc = self._ic_method_map[method]
            except KeyError as err:
                raise ValueError(
                    "'%s' is an invalid method. Must be one "
                    "of ('aic', 'aicc', 'bic', 'fixed')"
                ) from err

            fits = []
            icvals = []
            for lag_term in range(1, maxlag + 1):  # 1 -> maxlag (incl)
                try:
                    fit = self._fit_ocsb(x, m, lag_term, maxlag)
                    fits.append(fit)
                    icvals.append(icfunc(fit))
                except np.linalg.LinAlgError:  # Singular matrix
                    icvals.append(np.nan)
                    fits.append(None)

            # If they're all NaN, raise
            if np.isnan(icvals).all():
                raise ValueError(
                    "All lag values up to 'maxlag' produced "
                    "singular matrices. Consider using a longer "
                    "series, a different lag term or a different "
                    "test."
                )

            # Compute the information criterion vals
            best_index = int(np.nanargmin(icvals))
            maxlag = best_index - 1

            # Save this in case we can't compute a better one
            crit_regression = fits[best_index]

        # Compute the actual linear model used for determining the test stat
        try:
            regression = self._fit_ocsb(x, m, maxlag, maxlag)
        except np.linalg.LinAlgError as err:  # Singular matrix
            if crit_regression is not None:
                regression = crit_regression
            # Otherwise we have no solution to fall back on
            else:
                raise ValueError(
                    "Could not find a solution. Try a longer "
                    "series, different lag term, or a different "
                    "test."
                ) from err

        # Get the coefficients for the z4 and z5 matrices
        tvals = regression.tvalues[-2:]  # len 2
        return tvals[-1]  # just z5, like R does it

    def estimate_seasonal_differencing_term(self, x):
        """Estimate the seasonal differencing term.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        D : int
            The seasonal differencing term. For different values of ``m``,
            the OCSB statistic is compared to an estimated critical value, and
            returns 1 if the computed statistic is greater than the critical
            value, or 0 if not.
        """
        if not self._base_case(x):
            return 0

        # ensure vector
        x = check_endog(x, dtype=DTYPE, preserve_series=False)

        # Get the critical value for m
        stat = self._compute_test_statistic(x)
        crit_val = self._calc_ocsb_crit_val(self.m)
        return int(stat > crit_val)