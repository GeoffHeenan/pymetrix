// A stochastic growth model with persistent technology and government spending, correlated shocks and CRRA utility,
// for checking third-order solutions against Dynare. Written for pymetrix's tests (not part of Dynare); read by pymetrix
// with read_mod and run by Dynare in the dynare-reference workflow (case rbc_order3).
var c k y z g;
varexo e eg;
parameters alpha beta delta rho rhog gbar gamma;
alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho = 0.9;
rhog = 0.8;
gbar = 0.5;
gamma = 2;
model;
1 = beta*(c/c(+1))^gamma*(1 + alpha*exp(z(+1))*k^(alpha-1) - delta);
y = exp(z)*k(-1)^alpha;
k = y + (1-delta)*k(-1) - c - gbar*exp(g);
z = rho*z(-1) + e;
g = rhog*g(-1) + eg;
end;
steady_state_model;  // exact, so that both programs expand around the same point
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha - 1));
y = k^alpha;
c = y - delta*k - gbar;
z = 0;
g = 0;
end;
steady;
shocks;
var e; stderr 0.01;
var eg; stderr 0.02;
corr e, eg = 0.3;
end;
stoch_simul(order=3, irf=0, nograph, noprint);
