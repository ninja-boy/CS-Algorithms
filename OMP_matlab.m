clc; close all; clear all;
function x = omp(A, b, K)  % A is Theta matrix, b is observed signal vector, K is sparsity value

    % Normalize columns of A
    norms = vecnorm(A);
    A = A ./ norms;  % Normalizes all columns of A to unit L2 norm using vecnorm function

    % Remove duplicate columns, mostly not necessary
    % [A, ~, idxMap] = unique(A', 'rows');
    % A = A';

    r = b; % initially, residue is assumed to be equal to b
    Lambda = [];
    N = size(A, 2); 
    x = zeros(N, 1);

    for k = 1:K
    
        h_k = abs(A' * r); 
        h_k(Lambda) = 0; % ignore already selected indices
        [~, l_k] = max(h_k);
        
        Lambda = [Lambda, l_k];

        Asub = A(:, Lambda);
        x_sub = Asub \ b; 
        
        x = zeros(N, 1);
        
        % Rescale recovered values
        x(Lambda) = x_sub ./ norms(Lambda)';
        r = b - A .* norms * x;  % A .* norms = original A
        
    end
end
