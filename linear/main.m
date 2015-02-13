
%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('airfoil_self_noise.txt');
%Only above expression needs to be changed for different sets
[a,b] = size(data);
X = data(:, 1:b-1);
% and possibly this expression too
y = data(:, b);
m = length(y);



fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(b, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
pred = X*theta;

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('\nAccuracy: %f\n',100 - ((mean((abs(pred - y))./y))*100));

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');


%% Load Data
data = load('airfoil_self_noise.txt');
[a,b] = size(data);
X = data(:, 1:b-1);
y = data(:, b);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);
pred = X*theta;

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('\nAccuracy: %f\n',100 - ((mean((abs(pred - y))./y))*100));

