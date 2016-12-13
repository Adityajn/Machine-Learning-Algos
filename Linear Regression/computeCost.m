function J = computeCost(X, y, theta)

%========================================================================
%Compute cost for linear regression with multiple variables
%   J = computeCost(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%========================================================================


	m = length(y); % number of training examples


%=========================================================================
% X -> m * (n+1)			| first column 1 ,contains m training examples which has n variables 	
% theta -> (n+1) * 1 		| contains all variable  htheta(x) = theta0 + theta1 * x1 + ..... + thetan * xn
% y -> m*1					| its output for each one of the training examples
%=========================================================================

	predictions = X * theta;		% it is h(theta)
	error=sum((predictions-y).^2)	% is is error in prediction vs real outcome
	J = 1/(2*m) * error;			% it is computed cost    J = 1/2m summation( ( h(theta)-y )^2 )

end
