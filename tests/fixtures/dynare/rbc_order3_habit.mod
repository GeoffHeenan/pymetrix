// rbc_order3.mod with habits in consumption and technology following an AR(4), so that the solution has seven
// states, three of them auxiliary variables for the longer lags. For checking third-order solutions against Dynare.
// Written for pymetrix's tests (not part of Dynare); run by Dynare in the dynare-reference workflow (case rbc_order3).
var c k y z g lam;
varexo e eg;
parameters alpha beta delta rho1 rho2 rho3 rho4 rhog gbar gamma h;
alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho1 = 0.5;
rho2 = 0.2;
rho3 = 0.1;
rho4 = 0.05;
rhog = 0.8;
gbar = 0.5;
gamma = 2;
h = 0.6;
model;
lam = (c - h*c(-1))^(-gamma);
1 = beta*lam(+1)/lam*(1 + alpha*exp(z(+1))*k^(alpha-1) - delta);
y = exp(z)*k(-1)^alpha;
k = y + (1-delta)*k(-1) - c - gbar*exp(g);
z = rho1*z(-1) + rho2*z(-2) + rho3*z(-3) + rho4*z(-4) + e;
g = rhog*g(-1) + eg;
end;
steady_state_model;  // exact, so that both programs expand around the same point
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha - 1));
y = k^alpha;
c = y - delta*k - gbar;
lam = (c*(1 - h))^(-gamma);
z = 0;
g = 0;
end;
steady;
shocks;
var e; stderr 0.01;
var eg; stderr 0.02;
end;
stoch_simul(order=3, irf=0, nograph, noprint);
