function [g, k_dock] = spot_policy(obs)
%SPOT_POLICY  Full SPOT docking MLP policy: raw observation -> ICCBF gains.
%   obs   : 28x1 raw observation (same ordering as the Python env _obs()).
%   g     : 3x3 class-K gains  (rows = tar-KOZ, obs-KOZ, LOS ; cols = a0,a1,a2)
%   k_dock: scalar CLF decay rate.
%
%   Self-contained: weights + normalization + decode constants are in
%   spot_policy.mat. Pure matrix math -> clean Simulink code generation.
%   No importNetworkFromONNX / dlnetwork needed.

    persistent d
    if isempty(d)
        d = coder.load('spot_policy.mat');     % use load() if not generating code
    end

    % stage 2 -- VecNormalize (z-score + clip)
    o = min(max((obs(:) - d.obs_mean) ./ d.obs_std, -d.clip_obs), d.clip_obs);

    % stage 3 -- the trained MLP (obs -> action mean)
    h = tanh(d.W1 * o + d.b1);
    h = tanh(d.W2 * h + d.b2);
    a = d.W3 * h + d.b3;                        % 10x1 action

    % stage 4 -- residual decode (action -> gains)
    a = min(max(a, -1), 1);
    grid = reshape(a(1:9), [3, 3]).';          % .' REQUIRED: Python reshape is row-major
    g = d.BASE.' + d.BAND.' .* grid;           % rows tar,obs,los ; cols a0,a1,a2
    g(:, 1:2) = min(max(g(:, 1:2), 0), d.ACOEF_HI);
    g(:, 3)   = min(max(g(:, 3), d.HSLACK_LO), d.HSLACK_HI);
    k_dock = max(0.1, d.KDOCK + d.KBAND * a(10));
end
