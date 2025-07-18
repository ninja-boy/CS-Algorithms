clc; clear; close all;

n = 128;      % length of the signal    
m = 32;       % change this to adjust the number of measurements    
K = 5;        % number of sparse coefficients    
alpha = 0.05; % threshold for sparsity    
i = 50;       % number of iterations

z_true = zeros(n,1);
idx = randperm(n, K);
z_true(idx) = randn(K,1);

Psi = idct(eye(n));         
x = Psi * z_true;   % real signal

% Gaussian noise
snr = 10; 
s_p = norm(x)^2 / m;
n_p = s_p / (10^(snr/10));
noise = sqrt(n_p) * randn(m, 1);

Phi = randn(m,n);
Phi = Phi ./ vecnorm(Phi')'; 
%y = Phi * x;  
y = Phi * x + noise;

theta = Phi * Psi;  % Dictionary for CoD            

% Coordinate Descent Algorithm
 z = zeros(n,1);
B = theta' * y;  % Initial coefficients
S = eye(n) - theta' * theta;  % Residual matrix

mse_history = zeros(i,1); % Preallocate MSE array

for t = 1:i
    z_bar = sign(B) .* max(abs(B) - alpha, 0); % Thresholding step
    [~, k] = max(abs(z - z_bar));     % Find index of maximum change         

    delta = z_bar(k) - z(k);  % Update step
    B = B + S(:,k) * delta; % Update coefficients
    z(k) = z_bar(k); % Update sparse coefficient

    x_rec_iter = Psi * z; % Reconstruct at this iteration
    mse_history(t) = mean((x - x_rec_iter).^2); % Compute MSE
end

x_rec = Psi * z;    % recovered signal

figure;
plot(x, 'r--'); hold on;
plot(x_rec, 'b-');
legend('Original Signal', 'Reconstructed Signal');
title('Original vs Reconstructed Signal');
xlabel('Index');
ylabel('Amplitude');
grid on;

% Plot MSE vs Iterations
figure;
plot(1:i, mse_history, 'LineWidth', 2);
xlabel('Iteration');
ylabel('MSE');
title('MSE vs Iterations for CoD');


