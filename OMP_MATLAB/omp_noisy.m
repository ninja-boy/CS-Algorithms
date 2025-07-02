clc; close all; clear all;
function x = omp(A, b, K)
    % Store column norms and normalize A
    norms = vecnorm(A);
    A = A ./ norms;

    % Initialization
    r = b;
    Lambda = [];
    N = size(A, 2);
    x = zeros(N, 1);

    for k = 1:K
        proj = abs(A' * r);
        proj(Lambda) = 0;
        [~, lam] = max(proj);
        Lambda = [Lambda, lam];
        Asub = A(:, Lambda);
        x_sub = Asub \ b;
        x = zeros(N, 1);
        % Rescale recovered values
        x(Lambda) = x_sub ./ norms(Lambda)';
        r = b - A .* norms * x;  % A .* norms = original A
    end
end

% Problem dimensions
n = 256;      % signal length
m = 50;      % number of measurements
k = 5;       % sparsity level

% Generate a random sparse signal
x_true = zeros(n, 1);
support = randperm(n, k);
x_true(support) = randn(k, 1);

% Generate sensing matrix (Gaussian)
A = randn(m, n);

% Generate noiseless measurements
b_clean = A * x_true;

% Add Gaussian noise
snr_db = 20;  % desired SNR in dB
signal_power = norm(b_clean)^2 / m;
noise_power = signal_power / (10^(snr_db/10));
noise = sqrt(noise_power) * randn(m, 1);
b_noisy = b_clean + noise;

% Recover using OMP
x_rec = omp(A, b_noisy, k)

% Plot original vs recovered
figure;
stem(x_true, 'bo', 'DisplayName', 'Original'); hold on;
stem(x_rec, 'r--', 'DisplayName', 'Recovered');
legend;
xlabel('Index');
ylabel('Amplitude');
title(sprintf('OMP Recovery with Noise (SNR = %d dB)', snr_db));

% Errors
l2_err = norm(x_true - x_rec);
fprintf('SNR: %d dB\n', snr_db);
fprintf('Reconstruction error (L2 norm): %.4e\n', l2_err);
fprintf('Support recovery accuracy: %d / %d correct\n', ...
    sum((x_rec ~= 0) & (x_true ~= 0)), k);