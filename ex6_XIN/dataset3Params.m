function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.3;
sigma = 0.1;

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
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predictions = svmPredict(model, Xval);
pre_tmp = mean(double(predictions ~= yval));

C_tmp = [0.01, 0.03, 0.1, 0.3, 1, 3, 10,30];
sigma_tmp = [0.01, 0.03, 0.1, 0.3, 1, 3, 10,30];


for i=1:size(C_tmp,2)
	for j=1:size(sigma_tmp,2)
		model= svmTrain(X, y, C_tmp(i), @(x1, x2) gaussianKernel(x1, x2, sigma_tmp(j))); 
		predictions = svmPredict(model, Xval);
		tmp = mean(double(predictions ~= yval));
		if(tmp < pre_tmp)
			pre_tmp = tmp;
			C = C_tmp(i);
			sigma = sigma_tmp(j);
		end
	end
end



% =========================================================================

end
