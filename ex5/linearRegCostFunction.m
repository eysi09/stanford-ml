function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X*theta;
err = (h_theta - y);
J_unreg = (1/(2*m))*sum(err.^2);
% Not regularising theta_0
reg_term = (lambda/(2*m))*sum(theta(2:end).^2);
J = J_unreg + reg_term;

% =========================================================================

% Gradient

for j = 1:size(theta)(1)
  if j == 1 % Really j_zero, but ya know, indexing.
    grad(j) = (1/m)*sum(err.*X(:,j:j));
  else
    grad(j) = (1/m)*sum(err.*X(:,j:j)) + (lambda/m)*theta(j);
  end
end

grad = grad(:);

end
