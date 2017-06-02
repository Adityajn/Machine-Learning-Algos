function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%===================================================================================    
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = gradientDescent(x, y, theta, alpha, num_iters) 
%===================================================================================

    m = length(y);   % number of training examples

%=====================================================================================
% X -> m*(n+1)          |  first column 1 ,contains m training examples which has n variables
% theta -> (n+1) * 1    |  contains all variable  htheta(x) = theta0 + theta1 * x1 + ..... + thetan * xn
% y -> m*1              |  its output for each one of the training examples
% alpha -> 0.01         |  learning rate
% num_iters -> 400      |  number of iterations to be performed (can be changed as per necessity)
%=====================================================================================

    J_history = zeros(num_iters, 1);    % J_history saves all value of cost that have taken, 
                                        % can be used to plot graph and see if cost is still can decrese or not

    for iter = 1:num_iters

        theta = theta-((alpha/m)*(X'*(X*theta-y)));  % theta = theta - (alpha/m) summation( (htheta(x) - y)*x  )

        J_history(iter) = computeCost(X, y, theta);    % save to history,the value of current cost

    end

end
