function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of training set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set


%/home/raja/Education/sem 2/ML/projects/prj3/datasets
%newdataset_MLR.mat  newdataset_SVM.mat
clearvars;

load('/mnist_all.mat'); %%%%% FIXME -- need to change location
%load('.../newdataset_MLR.mat');
%load('.../newdataset_SVM.mat');

n_validation = 1000;

validation_data = [train0(1:n_validation, :); train1(1:n_validation, :); ...
    train2(1:n_validation, :); train3(1:n_validation, :); train4(1:n_validation, :);...
    train5(1:n_validation, :); train6(1:n_validation, :); train7(1:n_validation, :);
    train8(1:n_validation, :); train9(1:n_validation, :)];
validation_label = ones(size(train0(1:n_validation, :), 1), 1);
validation_label = [validation_label; 2 * ones(size(train1(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 3 * ones(size(train2(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 4 * ones(size(train3(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 5 * ones(size(train4(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 6 * ones(size(train5(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 7 * ones(size(train6(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 8 * ones(size(train7(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 9 * ones(size(train8(1:n_validation, :), 1), 1)];
validation_label = [validation_label; 10 * ones(size(train9(1:n_validation, :), 1), 1)];

train_data = [train0( n_validation + 1:end, :); train1( n_validation + 1:end, :); ...
    train2( n_validation + 1:end, :); train3( n_validation + 1:end, :);...
    train4( n_validation + 1:end, :); train5( n_validation + 1:end, :);...
    train6( n_validation + 1:end, :); train7( n_validation + 1:end, :);...
    train8( n_validation + 1:end, :); train9( n_validation + 1:end, :)];
train_label = ones(size(train0( n_validation + 1:end, :), 1), 1);
train_label = [train_label; 2 * ones(size(train1( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 3 * ones(size(train2( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 4 * ones(size(train3( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 5 * ones(size(train4( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 6 * ones(size(train5( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 7 * ones(size(train6( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 8 * ones(size(train7( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 9 * ones(size(train8( n_validation + 1:end, :), 1), 1)];
train_label = [train_label; 10 * ones(size(train9( n_validation + 1:end, :), 1), 1)];

test_data = [test0; test1; test2; test3; test4; test5; test6; test7; test8; test9];
test_label = ones(size(test0, 1), 1);
test_label = [test_label; 2 * ones(size(test1, 1), 1)];
test_label = [test_label; 3 * ones(size(test2, 1), 1)];
test_label = [test_label; 4 * ones(size(test3, 1), 1)];
test_label = [test_label; 5 * ones(size(test4, 1), 1)];
test_label = [test_label; 6 * ones(size(test5, 1), 1)];
test_label = [test_label; 7 * ones(size(test6, 1), 1)];
test_label = [test_label; 8 * ones(size(test7, 1), 1)];
test_label = [test_label; 9 * ones(size(test8, 1), 1)];
test_label = [test_label; 10 * ones(size(test9, 1), 1)];

%   Preprocess the data
train_data = double(train_data); % convert training data to matrix of double
validation_data = double(validation_data); % convert validation data to matrix of double
test_data = double(test_data);   % convert testing data to matrix of double

% get the number of training, validation and test examples
n_feature = size(test_data, 2);

%   Delete features which don't provide any useful information for
%   classifiers
sigma = std(train_data);
new_train_data = [];
new_validation_data = [];
new_test_data = [];
for i = 1 : n_feature
    if (sigma(i) > 0.001)
        new_train_data = [new_train_data train_data(:, i)];
        new_validation_data = [new_validation_data validation_data(:, i)];
        new_test_data = [new_test_data test_data(:, i)];
    end
end

train_data = new_train_data;
validation_data = new_validation_data;
test_data = new_test_data;

clear 'new_train_data' 'new_validation_data' 'new_test_data'

% scale data to 0 and 1
train_data = train_data / 255;
validation_data = validation_data / 255;
test_data = test_data / 255;

end

