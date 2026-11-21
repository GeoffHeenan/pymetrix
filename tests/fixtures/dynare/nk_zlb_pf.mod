// The same model with the floor written as max(), and the shocks known from period 1: Dynare's perfect-foresight
// solver, with the floor (-Dwithbound=true) and without it (-Dwithbound=false). With the shocks anticipated, OccBin's
// piecewise-linear path is this exact solution (the model is linear in each regime). Run by the dynare-reference
// workflow (case occbin_zlb).
var y pi i;
varexo ey epi ei;
parameters sigma beta kappa rho phi_pi phi_y i_floor;
sigma = 1; beta = 0.99; kappa = 0.1; rho = 0.8; phi_pi = 1.5; phi_y = 0.5; i_floor = -0.75;
model;
y = y(+1) - (1/sigma)*(i - pi(+1)) + ey;
pi = beta*pi(+1) + kappa*y + epi;
@#if withbound
i = max(rho*i(-1) + (1 - rho)*(phi_pi*pi + phi_y*y) + ei, i_floor);
@#else
i = rho*i(-1) + (1 - rho)*(phi_pi*pi + phi_y*y) + ei;
@#endif
end;
initval;
y = 0; pi = 0; i = 0;
end;
steady;
shocks;
var ey; periods 1:4; values -2.5;
end;
perfect_foresight_setup(periods=200);
perfect_foresight_solver;
