fit = Equation("lwage ~ educ + exper + expersq").fit(mroz)
fit_white = Equation("lwage ~ educ + exper + expersq").fit(mroz, method=OLS(cov="white"))
