// pymetrix's templates.new_keynesian(lower_bound=0) in deviations: the floor on the rate is (0 - 1 - 2)/4 = -0.75.
// OccBin (Dynare's occbin_solver) with a demand shock of -2.5 as a surprise in each of periods 1-4. pymetrix's own
// model file, run by the dynare-reference workflow (case occbin_zlb) for tests/fixtures/dynare-occbin-zlb.json.
var y pi i inot;
varexo ey epi ei;
parameters sigma beta kappa rho phi_pi phi_y i_floor;
sigma = 1; beta = 0.99; kappa = 0.1; rho = 0.8; phi_pi = 1.5; phi_y = 0.5; i_floor = -0.75;
model;
[name='IS'] y = y(+1) - (1/sigma)*(i - pi(+1)) + ey;
[name='Phillips'] pi = beta*pi(+1) + kappa*y + epi;
[name='notional'] inot = rho*i(-1) + (1 - rho)*(phi_pi*pi + phi_y*y) + ei;
[name='policy', relax='zlb'] i = inot;
[name='policy', bind='zlb'] i = i_floor;
end;
occbin_constraints;
name 'zlb'; bind inot <= i_floor; relax inot > i_floor;
end;
steady_state_model;
y = 0; pi = 0; i = 0; inot = 0;
end;
steady;
shocks(surprise);
var ey; periods 1:4; values -2.5;
end;
occbin_setup;
occbin_solver(simul_periods=20);
