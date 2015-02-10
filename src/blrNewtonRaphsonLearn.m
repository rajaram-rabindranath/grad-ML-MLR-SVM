function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
%blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method
% R = spdiags( R_nn, 0, numel(R_nn), numel(R_nn) ); could have used this as
% well

%% =========== adding bias column to the training data
%objFunction = @(params) blrObjFunction(params, train_data, T(:, i));
bias_col = ones(size(X,1),1);
X_ = [bias_col X];

%% ================ Compute the Weights ===================
w = initial_w;
for iter = 1 : n_iter 
    Y=sigmoid(w'*X_');                  %% 1 x N [row_vector]
    error_grad = (Y' - t)' * X_;        %% 1 x D [row_vector]
    R_nn = Y' .* (1-Y');                %% N X 1 [col_vector]
    R = diag(sparse(R_nn));             %% N X N [diagonal matrix]  
    H = X_' * R * X_;                   %% D x D [Hessian matrix]  
    w_new = w - (pinv(H)*error_grad');  %% D x 1 [col_vector] - weight vector
    w = w_new;                          %% -"-
end

end
