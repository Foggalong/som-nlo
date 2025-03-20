var x1 default -2;
var x2 default 2;
var x3 default 2;
var x4 default -1;
var x5 default -1;

minimize obj: x1*x2*x3*x4*x5;

subject to cons1: x1*x1+x2*x2+x3*x3+x4*x4+x5*x5 = 10;
subject to cons2: x2*x3 = 5*x4*x5;
subject to cons3: x1^3 + x2^3 = -1;

