clc; close all; clear all;
function x = omp(A, b, K)
    originalA = A;               % Store the original A
    norms = vecnorm(A);
    A = A ./ norms;

    r = b;
    Lambda = [];
    N = size(A, 2);
    x = zeros(N, 1);

    for k = 1:K
        h_k = abs(A' * r);
        h_k(Lambda) = 0;
        [~, l_k] = max(h_k);

        Lambda = [Lambda, l_k];
        Asub = A(:, Lambda);
        x_sub = Asub \ b;

        x = zeros(N, 1);
        x(Lambda) = x_sub ./ norms(Lambda)';
        r = b - originalA(:, Lambda) * x(Lambda);   % Corrected
    end
end



n = 256;     
m = 50;      
k = 2;        

freqs = [randi([1, 10]),randi([1, 10])];%,randi([10, 50]),randi([10, 50]),randi([10, 50])]               
x_freq = zeros(n, 1);              
x_freq(freqs) = [1; 1];%; 1; 1; 1];            
x_time = real(ifft(x_freq))*n;       

psi = randn(m,n);                  
b = psi * x_time;                     

phi = dftmtx(n);
Theta = psi * phi';                    

x_freq_rec = omp(Theta, b, k);      
x_rec = real(ifft(x_freq_rec))*n;    

figure;
t = 0:n-1;

% Plot the original and recovered signals smoothly
plot(t, x_time, 'b-'); hold on;
plot(t, x_rec, 'r--');
legend;
xlabel('Index');
ylabel('Amplitude');
title('OMP (sinusoidal)');
