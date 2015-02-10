function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function
% 
% W = initialWeights;
% X = train_data;
% size(T)
% size(train_data)
% size(W)

%% transform (D*C) x 1 vector TO D*C matrix
W = reshape(W, size(X, 2) + 1, size(T, 2)); 

%% =========== adding bias column to the training data
bias_col = ones(size(X,1),1);
X_ = [bias_col X];

%% ====== calculate ERROR = -ln p(T|w1,w2,.....wk) 
y_left = (W'*X_');                          %% C x N [matrix]
y_right = logsumexp(y_left,1);              %% 1 x N [row_vector]
y_right = repmat(y_right,size(W,2),1);      %% C x N [matrix]
y_log = y_left - y_right;                   %% C x N 
error = -1 * sum(sum((y_log .* T')));       %% 1 x 1 scalar   

%% ====== calculate ERROR GRADIENT ---
y_num = exp(W'*X_');
y_denom = sum(y_num);
y_denom = repmat(y_denom,size(W,2),1);
y = y_num ./ y_denom;
error_grad_matrix = X_'* (y' - T);
error_grad = reshape(error_grad_matrix,size(X_,2)*size(T,2),1);

end
