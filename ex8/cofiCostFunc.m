function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Note that we multiply R element wise to get rid of entries with no ratings
% Same as summing over X*Theta' for only those i's and j's where R(i,j) = 1
J = (1/2)*sum(sum(((X*Theta').*R - Y).^2));

reg_term = 0.5*lambda*sum(sum(Theta.^2)) + 0.5*lambda*sum(sum(X.^2));

J = J + reg_term;

X_grad_test = zeros(num_movies, num_features);
Theta_grad_test = zeros(num_users, num_features);

for i = 1:num_movies
  idx_m = find(R(i, :)==1);
  Theta_temp = Theta(idx_m, :);
  Y_temp = Y(i, idx_m);
  reg_term = lambda*X(i,:);
  X_grad(i, :) = (X(i, :)*Theta_temp' - Y_temp)*Theta_temp + reg_term;
end

% Naive summing (produces wrong results for the first row but correct
% results for the remaining rows. Not really sure why.)

% for i = 1:num_movies
%   for j = 1:num_users
%      if (R(i,j) != 0)
%        for k = 1:num_features
%          X_grad_test(i,k) = sum( (X(i,:)*Theta(j,:)' - Y(i,j))*Theta(j,k) );
%       end
%     end
%   end
% end

for j = 1:num_users
  idx_u = find(R(:,j)==1);
  X_temp = X(idx_u, :);
  Y_temp = Y(idx_u, j);
  reg_term = lambda*Theta(j,:);
  % Note that we need to transpose the term before the last X_temp
  % for dims to match although that doesn't really conform to the formula.
  Theta_grad(j, :) = (X_temp*Theta(j,:)' - Y_temp)'*X_temp + reg_term;
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
