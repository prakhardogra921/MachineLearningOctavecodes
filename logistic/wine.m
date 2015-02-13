%% Initialization
clear ; close all; clc


input_layer_size  = 13;  
num_labels = 3;          

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading Data ...\n');

data = load('wine2.txt');
X = data(:,2:14);
y = data(:,1);
m = size(X, 1);
pause;

%% ============ Part 2: Vectorize Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')
fprintf('Lambda = 0\n')
lambda = 0;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


pause;

%% ============ Part 2: Vectorize Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')
fprintf('Lambda = 0.5\n')
lambda = 0.5;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


pause;
%% ============ Part 2: Vectorize Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')
fprintf('Lambda = 1\n')
lambda = 1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


pause;
%% ============ Part 2: Vectorize Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')
fprintf('Lambda = 2\n')
lambda = 2;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


pause;
