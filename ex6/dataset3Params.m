function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

testVals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
l = length(testVals);

errorMat = zeros(l,l);

for i = 1:l
  for j = 1:l
    CTestval = testVals(i);
    sigmaTestVal = testVals(j);
    model = svmTrain(X, y, CTestval, ...
      @(x1, x2) gaussianKernel(x1, x2, sigmaTestVal));
    predictions = svmPredict(model, Xval);
    predictionErr = mean(double(predictions ~= yval));
    errorMat(i,j) = predictionErr;
  end
end

% Find index of min value
[M,I] = min(errorMat(:));
[I_row, I_col] = ind2sub(size(errorMat),I);

C = testVals(I_row);
sigma = testVals(I_col);

% =========================================================================

end
