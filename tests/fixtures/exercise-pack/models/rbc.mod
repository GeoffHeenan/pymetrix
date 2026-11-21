// A stochastic growth model, in Dynare's syntax: for tests/test_exercises.py
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
shocks;
var e; stderr 0.01;
end;
