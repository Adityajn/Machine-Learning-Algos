function [J, grad] = costFunctionLogisticReg(theta, X, y, lambda)

%======================================================================================
%   require sigmoid function 
%	COSTFUNCTIONLINEARREG Compute cost and gradient for logistic regression with regularization
%   J = costFunctionLinearReg(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%======================================================================================

	m = length(y); % number of training examples
	J = 0;
	grad = zeros(size(theta));

%=======================================================================================
% theta -> (n+1)*1
% X -> m*(n+1)
% y -> m*1
% J -> 1*1							regularized cost
%grad ->(n+1)*1 					d/dtheta(cost function)
% lambda -> 1  						regularization parameter
%========================================================================================

	h = X * theta;				% m*1 	prediction  h(x) ->X*theta

	thetan=theta(2:end,:);		% n*1 	theta after removing 1st columns i.e. 1's
	Xn= X(:,2:end);				% m*n 	X after removing 1st row

	J = 1/(2*m) *sum((h-y).^2) + (lambda/(2*m))*sum(thetan.^2);		
																
																%J -> Cost = -1/m ( y*log(h(x)) + (1-y)*log(1-h(x)) )
	grad(1,:) = ( X(:,1)' * (h-y) )./m;		% 1st row of gradient

	grad(2:end,:) = (Xn' * (h-y))./m + (lambda/m)*thetan;	%rest of the row in gradient



end
