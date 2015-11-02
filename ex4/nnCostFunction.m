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

% Part 1

% Initialize the (capital) Deltas
Delta1 = 0;
Delta2 = 0;

for i = 1:m
   % FORWARD PROP

   % First we find h_theta (in K dimensions)
   a1 = [1; X(i,:)']; % bias already added above
   z2 = Theta1*a1;
   a2 = [1; sigmoid(z2)]; % add bias
   z3 = Theta2*a2; % no need to transpose here since a2 is already a colvec
   a3 = sigmoid(z3);
   h_Theta = a3;
   % Then we compute the cost (using vectorization)
   y_vec = (1:num_labels == y(i))'; % create logical array from y val
   cost = -y_vec'*log(h_Theta) - (1 - y_vec')*log(1-h_Theta);
   J = J+cost;
   
   % BACK PROP
   d3 = a3 - y_vec;
   d2 = (Theta2(:,2:end)'*d3).*sigmoidGradient(z2); % Add the bias to z2
   
   Delta2 = Delta2 + d3*a2';
   Delta1 = Delta1 + d2*a1';
end

J = J/m;

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% Part 3 Regularize cost function

temp_Theta1 = Theta1(:,2:end); % remove bias column
temp_Theta2 = Theta2(:,2:end);

temp_Theta1_vec = reshape(temp_Theta1, size(temp_Theta1, 1) * size(temp_Theta1, 2), 1); % convert to vector for summing
temp_Theta2_vec = reshape(temp_Theta2, size(temp_Theta2, 1) * size(temp_Theta2, 2), 1);

reg_cost = (lambda/(2*m))*(sum(temp_Theta1_vec.^2) + sum(temp_Theta2_vec.^2));

J = J + reg_cost;

% -------------------------------------------------------------

% =========================================================================

% Regularize gradient
reg_grad_Theta1 = (lambda/m)*Theta1(:,2:end);
reg_grad_Theta2 = (lambda/m)*Theta2(:,2:end);
Theta1_grad = [Theta1_grad(:,1), Theta1_grad(:,2:end) + reg_grad_Theta1];
Theta2_grad = [Theta2_grad(:,1), Theta2_grad(:,2:end) + reg_grad_Theta2];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
