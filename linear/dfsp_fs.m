%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data

data = load('afsn.txt');

X = data(:,1:5);
y = data(:,6);
m = length(y);
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\nFirst applying Mean Normalization and then applying Feature Scaling\n');

%[X mu sigma] = featureNormalize(X);
Xn = featureNormalize(X);


X = [Xn(:,1)/10000 Xn(:,2)/15.4 Xn(:,3)/0.1524 Xn(:,4)/39.6 Xn(:,5)/0.03];

% Add intercept term to X
X = [ones(m, 1) Xn];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.001;
num_iters = 3000;

% Init Theta and Run Gradient Descent 
theta = zeros(6, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('Solving with normal equations...\n');

X = data(:,1:5);
y = data(:,6);

m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

