function [xT, transform] = elastic_normalize_transform(x, setPts)
% setPts is a list of percentiles that will be exactly pinned to the
% corresponding point in the normal distribution. All points between the
% setPts are linearly interpolated.

if nargin < 2
   setPts = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99];
end

N = sum(~isnan(x));
xP = prctile(x, [0, setPts, 100]);
setPts = [50/N, setPts, 100-50/N];
y = norminv(setPts/100);
transform = griddedInterpolant(xP, y, 'linear', 'linear');
xT = transform(x);
end

