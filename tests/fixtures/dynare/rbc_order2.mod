// A stochastic growth model with persistent technology, for checking second-order solutions against Dynare.
// Written for pymetrix's tests (not part of Dynare); read by pymetrix with read_mod and run by Dynare in the
// dynare-reference workflow (case rbc_order2).
var c k y z;
varexo e;
parameters alpha beta delta rho;
alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho = 0.9;
model;
1/c = beta/c(+1)*(1 + alpha*exp(z(+1))*k^(alpha-1) - delta);
y = exp(z)*k(-1)^alpha;
k = y + (1-delta)*k(-1) - c;
z = rho*z(-1) + e;
end;
steady_state_model;  // exact, so that both programs expand around the same point
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha - 1));
y = k^alpha;
c = y - delta*k;
z = 0;
end;
steady;
shocks;
var e; stderr 0.01;
end;
stoch_simul(order=2, irf=0, nograph, noprint);
