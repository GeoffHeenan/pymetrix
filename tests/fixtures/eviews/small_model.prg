' A small made-up macro model in EViews's program syntax, written for pymetrix's tests (pymetrix's own work,
' BSD-3-Clause): a model object with appended equations, lags, dlog, @pch, @trend, a comparison with the time
' variable t, an identity and an add factor.
model _small
_small.append cap = pk * k(-1)
_small.append ur = q / cap
_small.append q + m = fd + x
_small.append i / k(-1) = 0.9 * i(-1) / k(-2) - 0.002 * (ur - ur_star) / ur + 0.004 + ec_i
_small.append k = k(-1) * (1 - depr) + i
_small.append log(prod) = 9.5 + 0.01 * @trend(2000q1) + 0.02 * (t - t1) * (t < t1)
_small.append dlog(le) = 0.3 * dlog(q / prod) + 0.2 * log(q(-1) / prod(-1) / le(-1)) + ec_le
_small.append @identity rhi = wr * le + r_rhiq * q
_small.append co = (1 - sr) * rhi
_small.append fd = co + i + gd
_small.append m = 0.25 * fd * (1 + 0.1 * @pch(ur)) + ec_m
_small.append x = 0.3 * wd + ec_x
