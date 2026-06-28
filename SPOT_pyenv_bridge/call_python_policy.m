function [g, k_dock, status_code] = call_python_policy(x_red, x_black, x_obstacle, holding_radius)
%CALL_PYTHON_POLICY  SPOT RL class-K gain policy via pyenv co-execution.
%   Mirror of call_python_mpc.m. Each control step, pass the lab-frame states and
%   get back the class-K gains the RL policy chose -- your CBF-CLF QP uses them.
%
%   x_red, x_black, x_obstacle : 6-vectors [x y theta dx dy dtheta]
%   holding_radius             : current target keep-out size (scalar or [a b])
%   g        : 3x3 class-K gains (rows tar-KOZ, obs-KOZ, LOS ; cols a0,a1,a2)
%   k_dock   : scalar CLF decay rate
%   status_code : 1 ok, -99 on failure (g falls back to the nominal 1,0.5,0.25)

    persistent mod np
    if isempty(mod)
        np  = py.importlib.import_module("numpy");
        mod = py.importlib.import_module("spot_rl_policy");
    end

    try
        x_red      = reshape(double(x_red), 1, []);
        x_black    = reshape(double(x_black), 1, []);
        x_obstacle = reshape(double(x_obstacle), 1, []);
        r_hold     = reshape(double(holding_radius), 1, []);

        out = mod.get_gains( ...
            np.array(x_red,      pyargs("dtype", np.float64)), ...
            np.array(x_black,    pyargs("dtype", np.float64)), ...
            np.array(x_obstacle, pyargs("dtype", np.float64)), ...
            np.array(r_hold,     pyargs("dtype", np.float64)) );

        g = double(out{'gains'});              % 3x3 (rows tar,obs,los ; cols a0,a1,a2)
        k_dock = double(out{'k_dock'});
        status_code = 1;

    catch ME
        fprintf("\n===== SPOT RL policy call FAILED: %s =====\n", ME.message);
        g = [1 0.5 0.25; 1 0.5 0.25; 1 0.5 0.25];   % nominal fallback
        k_dock = 5.0;
        status_code = -99;
    end
end
