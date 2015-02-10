function [error, error_grad] = blrObjFunction(w, X, t)
% blrObjFunction computes 2-class Logistic Regression error function and
% its gradient.
%
% Input:
% w: the weight vector of size (D + 1) x 1 
% X: the data matrix of size N x D
% t: the label vector of size N x 1 where each entry can be either 0 or 1
%    representing the label of corresponding feature vector
% 
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size (D+1) x 1 representing the gradient of
%             error function



%% =========== adding bias column to the training data
%objFunction = @(params) blrObjFunction(params, train_data, T(:, i));
bias_col = ones(size(X,1),1);
X_ = [bias_col X];

%% ======================= sigmoid(w'X) ==========================
w_trans = w';                   %% 1 x (D+1) [row_vector]
X_trans = X_';                  %% (D+1) x N [matrix]
y=sigmoid(w_trans*X_trans);     %% 1 x N     [row_vector]

%% ================ ERROR = E(w) = -ln p(t|w) ===================
t_compliment = 1 - t;                              %% N x 1 [col_vector]
y_compliment = 1 - y;                              %% 1 x N [row_vector]

error_partA = log(y) * t;                          %% 1x1 [scalar]
error_partB = log(y_compliment) * t_compliment;    %% 1x1 [scalar]
error = sum(error_partA+error_partB) *-1;          %% 1x1 [scalar]

%% ================= ERROR GRADIENT = (y' - t) * x (D+1) x 1 vector ==========
error_grad =  X_trans * (y'-t);

end

