function [label] = blrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10. Each column is the weight
%    vector of a Logistic Regression classifier.
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%% =========== adding bias column to the training data
bias_col = ones(size(X,1),1);
X_ = [bias_col X];
%% ===================== make predictons
predictions = sigmoid(W' * X_');
[max_prob,label_trans]=max(predictions);
label = label_trans';
end

