%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: E:\mlclass-ex1\Folds5x2_pp.xlsx Worksheet: Sheet1
%
% To extend the code for use with different selected data or a different
% spreadsheet, generate a function instead of a script.

% Auto-generated by MATLAB on 2014/11/28 19:34:44

%% Import the data
[~, ~, raw] = xlsread('E:\mlclass-ex1\Folds5x2_pp.xlsx','Sheet1','A2:E9569');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
Folds5x2pp = reshape([raw{:}],size(raw));

%% Clear temporary variables
clearvars raw R;