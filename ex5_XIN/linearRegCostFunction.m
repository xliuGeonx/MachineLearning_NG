function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J = 0.5* sum((X*theta - y).^2) / m + 0.5*lambda*sum(theta(2:end).^2)/m;


grad(1) = sum((X(:,1:n)*theta-y).*X(:,1))/m;

for iter = 2:n

	grad(iter) = sum((X(:,1:n)*theta-y).*X(:,iter))/m + lambda / m * theta(iter);

end



% =========================================================================

grad = grad(:);

end