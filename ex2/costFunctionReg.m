function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% For the first term we borrow the solution from costFunction.m

h_theta = sigmoid(theta'*X')';

err_term = (1/m)*sum(-y'*log(h_theta) - (1-y')*log(1-h_theta));
reg_term = (lambda/(2*m))*sum(theta(2:end).^2);
J = err_term + reg_term;

% Grad for j = 0 (gives index 1 due to Octave indexing)
grad(1) = (1/m)*sum((h_theta - y)'*X(:,1));

for i = 2:size(theta)(1)
 grad(i) = (1/m)*sum((h_theta - y)'*X(:,i)) + (lambda/m)*theta(i);
end

% NOTE FROM EYSI: The above can (and probably should) be achived by the following:

% h_theta = sigmoid(X*theta);
% err_term = (1/m)*sum(-y.*log(h_theta) - (1 - y).*log(1 - h_theta));
% reg_term = (lambda/(2*m))*sum(theta(2:end).^2);
% J = err_term + reg_term;

% Calc grad without reg term:

% grad = (1/m)*X'*(h_theta - y);
% temp = theta;
% temp(1) = 0;

% Add reg term with theta(1) set to zero so it cancels out:

% grad = grad + (lambda/m).*temp;

% =============================================================

end
