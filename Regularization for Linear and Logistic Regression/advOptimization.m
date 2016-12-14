function [theta,cost] = advOptimization(initial_theta,X,y)

%====================================================================================
% require costFunction or costFunctionLogisticReg or costFunctionLinearReg

% Do optimization of theta with fminunc meaning f minimum unconstrained which is one
% of the advanced optimization techinique

% Can be used for linear or logistic regression
%====================================================================================

	options = optimset('GradObj', 'on', 'MaxIter', 400);	% initialize options

%====================================================================================
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost

% X-> m*(n+1) 				training set
% initial_theta -> (n+1)*1 	theta before initialization
% y > m*1 					output set

% here we are using costFunction change to costFunctionReg if u want regularization
%====================================================================================
	[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

	%[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, y)), initial_theta, options);
end