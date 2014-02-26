function [xT, transform, trans_description, test_stat] = normalize_transform(x)
% Normalizes data by choosing: a) a nonlinear transform from: identity,
% sqrt or log; and b) a linear transform to zero mean and 1 std.
% 
% The nonlinear transform is determined by choosing the one with the lowest
% (best) Lilliefors test statistic.

xMin = min(x);

% Test the untransformed data
[~, ~, idL, ~] = lillietest(x);

% Test the sqrt transformed data
if xMin < 0
   [~, ~, sqrtL, ~] = lillietest(sqrt(x - xMin));
else
   [~, ~, sqrtL, ~] = lillietest(sqrt(x));
end

% Test the log transformed data
if xMin > 0
   [~, ~, logL, ~] = lillietest(log(x));
else
   [~, ~, logL, ~] = lillietest(log(x - xMin + 1));
end

% Choose the transform with lowest test statistic, apply to data and then 
% linearly transform to mean = 0 and std = 1
[test_stat, idx] = min([idL, sqrtL, logL]);
switch idx
   case 1
      f = @(x) x;
      s = 'id';
   case 2
      if xMin < 0
         s = 'sqrt shift';
         f = @(x) sqrt(x - xMin);
      else
         s = 'sqrt';
         f = @sqrt;
      end
   case 3
      if xMin > 0
         s = 'log';
         f = @log;
      else
         s = 'log shift';
         f = @(x) log(x - xMin + 1);
      end
end

xT = f(x);
mu = mean(xT);
sigma = std(xT);
xT = (xT - mu)/sigma;
transform = @(x) (f(x) - mu)/sigma;
trans_description = sprintf('f: %s, mu: %.3f, sigma: %.3f', s, mu, sigma);
end

