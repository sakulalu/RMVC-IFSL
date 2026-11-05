%% Dataset: anchor, Dimension, lambda, mu, ACC, nmi, Purity, Fscore
%% MSRC(210) : 7 42 0.0001 10 0.8190 0.6998 0.8190 0.7019
%% Dermatology(358): 18 12 0.1 10 0.9413 0.8856 0.9413 0.9144
%% uci-digit(2000): 20 20 0.001 10 0.9135 0.8387 0.9135 0.8366
%% Caltech(2386): 20 40 1 1 0.7083 0.6845 0.7883 0.7430
%% mnist4(4000): 12 12 1 10 0.9023 0.7568 0.9023 0.8277
%% CCV(6773): 20 40 0.0001 10 0.2163 0.1716 0.2426 0.1305
%% Caltech101-all(9144): 102 204 0.0001 10 0.3173 0.4207 0.3818 0.2139
%% animals(10158): 50 100 0.0001 10 0.6259 0.6950 0.6506 0.5352

%% run_RMVC_IFSL_XMediaNet_clean.m

clear;
clc;
warning off;
addpath(genpath('./'));

%% Dataset
dataName = 'Dermatology.mat';
dataPath = './Dataset/';

%% Fixed parameters
anchor = 18;
d = 12;
lambda = 0.1;
mu = 10;

%% Run setting
num_runs = 1;
base_seed = 2026;

%% Load data
dataFile = fullfile(dataPath, dataName);
load(dataFile);

%% Assemble X and Y
if exist('X', 'var')
    X_data = X;
elseif exist('Img', 'var') && exist('Txt', 'var')
    X_data = cell(1, 2);
    X_data{1} = Img;
    X_data{2} = Txt;
else
    error('Cannot find X or Img/Txt in the dataset.');
end

if exist('Y', 'var')
    Y_data = Y;
elseif exist('label', 'var')
    Y_data = label;
else
    error('Cannot find Y or label in the dataset.');
end

Y_data = Y_data(:);
[~, ~, Y_data] = unique(Y_data);

n = length(Y_data);
num_view = length(X_data);
num_class = length(unique(Y_data));

%% Make sure each view is samples x features
for v = 1:num_view

    if ~isa(X_data{v}, 'double')
        X_data{v} = double(X_data{v});
    end

    if size(X_data{v}, 1) == n
        % already samples x features
    elseif size(X_data{v}, 2) == n
        X_data{v} = X_data{v}';
    else
        error('View %d size does not match the number of samples.', v);
    end

end

fprintf('Dataset: %s\n', dataName);
fprintf('Samples: %d, Views: %d, Clusters: %d\n', n, num_view, num_class);
fprintf('Parameters: anchor=%d, d=%d, lambda=%.6g, mu=%.6g\n', ...
    anchor, d, lambda, mu);

%% Run experiments
%% Random seed
rng(11, 'twister');

%% Run experiment
fprintf('\nRunning experiment with seed = 11\n');

tic;
[U, A, C, D, G, F, obj, pred_label] = double_self_paced_with_rank( ...
    X_data, Y_data, anchor, d, lambda, mu);
elapsed_time = toc;

%% Evaluation
res = Clustering8Measure(Y_data, pred_label);

fprintf('\nResults on %s\n', dataName);
fprintf('ACC     = %.4f\n', res(1));
fprintf('NMI     = %.4f\n', res(2));
fprintf('Purity  = %.4f\n', res(3));
fprintf('Fscore  = %.4f\n', res(4));
