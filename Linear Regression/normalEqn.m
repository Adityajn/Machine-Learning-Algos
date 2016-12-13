function [theta] = normalEqn(X, y)

%====================================================================
%NORMALEQN Computes the closed-form solution to linear regression 
%   normalEqn(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% use when n<10000, as inversion of matrix has complexity O(n^3).
% will slow the process. So its better to switch to iterative one (gradient descent)

%====================================================================


	theta = zeros(size(X, 2), 1);		%initialize theta as zeros (n+1)*1
										% X = m*(n+1)   y=m*1

	theta= pinv(X'*X)*X'*y				% theta is defined as ( X'*X )^-1 * X' * y

end
