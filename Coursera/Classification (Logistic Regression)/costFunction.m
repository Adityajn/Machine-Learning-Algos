function [J, grad] = costFunction(theta, X, y)

%=========================================================================
%   COSTFUNCTION Compute cost and gradient for logistic regression
%   J = costFunction(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%=========================================================================

	m = length(y); 										% number of training examples 
	J = 0;
	grad = zeros(size(theta));

%=========================================================================
% theta -> (n+1)*1 	contains all variable  htheta(x) = theta0 + theta1 * x1 + ..... + thetan * xn		
% X -> m*(n+1)		training sets 
% y -> m*1			training results		
% h -> m*1  		prediction
% J -> 1*1 			Cost Function
% grad -> (n+1)*1   d/dtheta (costFunction)
%=========================================================================

	h = sigmoid(X* theta); 								% h(x) -> sigmoid(X*theta)	 	(m*1)

	J = -1/m * ( y' * log( h ) + (1-y)'*log(1 - h) ) 	% J -> Cost = -1/m ( y*log(h(x)) + (1-y)*log(1-h(x)) )
 
 	grad = ( X' * (h-y) )./m; 							% grad -> (n+1)*1 = 1/m * ( h(x) - y )*X 

end
