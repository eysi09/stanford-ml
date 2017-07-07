function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% TODO Course ended before I managed to submit.
% Next up is 2.4 Gradient Checking.

computed_j = 0;

D1 = zeros(hidden_layer_size, input_layer_size + 1);
D2 = zeros(num_labels, hidden_layer_size + 1);

for t = 1:m

  % PART 1 - Feed forward and cost function
  x = X(t,:)'; % Note the transposition, the input is column vector
  a1 = x;
  a1_biased = [1; a1];
  z2 = Theta1*a1_biased;
  a2 = sigmoid(z2);
  a2_biased = [1; a2];
  z3 = Theta2*a2_biased;
  a3 = sigmoid(z3);
  h_theta = a3;

  % Calculate cost
  y_vec = zeros(num_labels, 1);
  y_vec(y(t)) = 1;

  inner_sum = sum(-y_vec.*log(h_theta) - (1 - y_vec).*log(1 - h_theta));

  computed_j += inner_sum;

  % PART 2 - Implement 4 step algorithm
  % Step 1 (computed above)
  % Step 2
  delta3 = a3 - y_vec;
  % Step 3
  delta2 = Theta2'*delta3.*sigmoidGradient([1; z2]);
  % Step 4
  D2 = D2 + delta3*a2_biased';
  D1 = D1 + delta2(2:end)*a1_biased';
end

% PART 1 - CONTINUED
% Regularization - Naive approach
% Note that (sum(Arr(:)) gives the double summation over Arr

Theta1_sq = Theta1.^2;
Theta2_sq = Theta2.^2;

% Naive approach:

% Theta1_reg = 0;
%
% for i = 1:size(Theta1_sq)(1)
%   inner_sum = sum(Theta1_sq(i,2:end));
%   Theta1_reg += inner_sum;
% end

% One liner approach:

Theta1_reg = sum(sum(Theta1_sq)(2:size(Theta1_sq)(2)));
Theta2_reg = sum(sum(Theta2_sq)(2:size(Theta2_sq)(2)));

reg_term = (lambda/(2*m))*(Theta1_reg + Theta2_reg);

J = (1/m)*computed_j + reg_term;

% PART 2 - CONTINUED

Theta1_grad = D1/m;
Theta2_grad = D2/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
