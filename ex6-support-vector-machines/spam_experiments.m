%% Initialization
clear ; close all; clc

% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');

filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');