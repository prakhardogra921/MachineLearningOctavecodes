fprintf('Solving with normal equations...\n');

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data

X = Folds5x2pp(:,1:4);
y = Folds5x2pp(:,5);

m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');