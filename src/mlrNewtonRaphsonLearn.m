function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size (D+1) x 10 represents the initial weight matrix
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W = zeros(size(X, 2) + 1, 10); % dummy return

% Testing
% X = train_data;
% initial_W = initialWeights;
% Testing End

n_features = size(X, 2) + 1;
n_class = size(T, 2);
W = initial_W;

N = size(X,1);
x = [ones(N,1),X];
%
% I = eye(n_features);
%
% for l = 1:n_iter
%     A = exp(x*w);
%     sum_A = sum(A,2);
%
%     sum_A_k = repmat(sum_A, 1, n_class);
%     y = A./sum_A_k;
%     for j = 1:n_class
%         H = zeros(n_features);
%         for k = 1:n_class
%             for i = 1:N
%                 H = H + y(i,k)*(I(k,j) - y(i,j))*x(i,:)'*x(i,:);
%             end
%         end
%         error_grad = x'*(y-T);
%         w(:,j) = w(:,j) - pinv(H)*(error_grad(:,j));
%     end
% end

% Test
I = eye(n_features);
% errors = zeros(n_iter, 1);
for l = 1:n_iter
    w = reshape(W, n_features, n_class);
    
    %     A = exp(x*w);
    %     sum_A = sum(A,2);
    %     sum_A_k = repmat(sum_A, 1, n_class);
    %     y = A./sum_A_k;
    
    a = x*w;
    logsum_a = logsumexp(a,2);
    y = exp(a - repmat(logsum_a, 1, n_class));
    
    error_grad = x'*(y-T);
    error_grad = reshape(error_grad, n_features*n_class, 1);
    H = cell(n_class);
    
    for j = 1:n_class
        for k = 1:n_class
            Rnn = y(:,k).*(I(k,j)-y(:,j));
            R = diag(sparse(Rnn));
            H{j,k} = x'*R*x;
        end
    end
    H = cell2mat(H);
    H = H + diag(sparse(repmat(10^-6,n_features*n_class,1)));
%     errors(l) = -sum(dot(T,(a - repmat(logsum_a, 1, n_class))));
    W = W - pinv(H)*error_grad;
    fprintf('Done iter %d\n',l);
end

% figure;
% plot(errors);

% Test End

end

