function [g, k_dock] = spot_rl_fcn(x_red, x_black, x_obstacle, holding_radius)
%#codegen
% Body of the Simulink MATLAB Function block that calls the RL gain policy.
% Mirror of the trmpc fcn.m. Python co-execution (py.*) CANNOT be code-generated,
% so the Python-calling wrapper is declared EXTRINSIC and every output is
% pre-allocated with a fixed size/type (codegen can't infer it from an mxArray).

coder.extrinsic('call_python_policy');

% --- outputs: declare concrete size/type FIRST (required for extrinsic calls) ---
g      = zeros(3, 3);     % class-K gains (rows tar,obs,los ; cols a0,a1,a2)
k_dock = 5.0;             % CLF decay rate

% --- typed temporaries to receive the extrinsic result ---
g_tmp      = zeros(3, 3);
k_tmp      = 0;
status_tmp = 0;

[g_tmp, k_tmp, status_tmp] = call_python_policy(x_red, x_black, x_obstacle, holding_radius);

g      = g_tmp;
k_dock = k_tmp;
% status_tmp (1 ok / -99 fail) available if you want to route it out too
end
