function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

z = -1 * z;
a = zeros(size(z));
a = exp(z);
a = a + 1;
f = ones(size(z));
g = f./ a;

% =============================================================

end
