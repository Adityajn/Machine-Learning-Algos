function [X_norm, mu, sigma] = featureNormalize(X)

%=======================================================================
%  FEATURENORMALIZE Normalizes the features in X 
%   featureNormalize(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%=======================================================================

	X_norm = X;						% X -> m*(n+1)

	mu = zeros(1,size(X, 2));  		% mu->1*(n+1) initialize mean Vector with zeroes with columns = no of variables and rows=1

	sigma = zeros(1, size(X, 2));	% sigma-> 1*(n+1) initialize standard deviation vector with zeros with columns = no of variables and rows=1

	mu=mean(X)+mu;					% mean(X)-> 1*(n+1) use mean function to compute mean and element wise addition with mean vector
	sigma=std(X)+sigma;				% std(X)-> 1*(n+1) use std function to compute standard deviation and element wise addition with std vector

	X_norm=(X-mu)./sigma;			% Noramlize, each row of X is subtarcted with mu vector and then divided with sigma vector


end
