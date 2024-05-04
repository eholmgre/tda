syms x y z xd yd zd real

X = [x y z xd yd zd];

rho = sqrt(x ^ 2 + y ^ 2 + z ^ 2);

F = [atan2(y, x) acos(z / rho) rho];

jacobian(F, X)
