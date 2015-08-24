function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

theta_0 = theta(1);
theta_1 = theta(2);
var_arr = (theta_0 + theta_1*X(:,2) - y).^2;
J = sum(var_arr)/(2*m);



% =========================================================================

end
