clear all
figure()
[z1, z2] = meshgrid(-10:.1:10, -10:.1:10);
y1 = 1./(1 + exp(-z1));
y2 = 1./(1 + exp(-z2));

F1 = (y1.^(4/3).*y2.^(2/3)).^(1/2);
surf(z1, z2, F1)
%contourf(z1, z2, F1, 11, 'ShowText', 'on')

figure()
y1 = min(0, z1);
y2 = min(0, z2);
F2 = exp((4*y1/3 + 2*y2/3)/2);
%contourf(z1, z2, F2, 11, 'ShowText', 'on')
surf(z1, z2, F2)
