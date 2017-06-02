function g = sigmoid(z)

%==========================================================
%SIGMOID Compute sigmoid functoon
%   J = sigmoid(z) computes the sigmoid of z.
%==========================================================

% z -> input vector of set

	g = zeros( size(z) );				% g is h(x) = prediction (may be for training set or new set)
	
	g = 1 ./ (1 + e .^ (-1 .* z) )		% g = 1/1+e^-z

end
