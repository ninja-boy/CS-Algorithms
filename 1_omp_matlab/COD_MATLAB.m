clc; clear; close all;

n = 128;          
m = 32;           
K = 5;            
alpha = 0.05;      
i = 50;      

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

theta = Phi * Psi;            

% Coordinate Descent Algorithm
z = zeros(n,1);
B = theta' * y;
S = eye(n) - theta' * theta;

for t = 1:i
    z_bar = sign(B) .* max(abs(B) - alpha, 0); 
    [~, k] = max(abs(z - z_bar));              

    delta = z_bar(k) - z(k);
    B = B + S(:,k) * delta;
    z(k) = z_bar(k);
end

x_rec = Psi * z;    % recovered ignal

figure;
plot(x, 'r--'); hold on;
plot(x_rec, 'b-');
legend('Original Signal', 'Reconstructed Signal');
title('Original vs Reconstructed Signal');
xlabel('Index');
ylabel('Amplitude');
grid on;


