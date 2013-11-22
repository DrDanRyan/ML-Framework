clear all
figure()
[z1, z2] = meshgrid(-20:.5:20, -20:.5:20);
y1 = max(0, z1);
y2 = max(0, z2);

F1 = exp(.5*(log(y1) + log(y2)));
surf(z1, z2, F1)
%contourf(z1, z2, F1, 41, 'ShowText', 'on')

% figure()
% y1 = min(0, z1);
% y2 = min(0, z2);
% F2 = exp((4*y1/3 + 2*y2/3)/2);
% %contourf(z1, z2, F2, 11, 'ShowText', 'on')
% surf(z1, z2, F2)
