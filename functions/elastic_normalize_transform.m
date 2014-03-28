function [xT, transform] = elastic_normalize_transform(x, varargin)
% Computes a monotone normalization function and applies it to the data.
% 
% Tie adjusted ranks for the data are computed and anchor points are determined.
% These anchor points are then mapped so that the CDF of the empirical
% distribution of the data matches the CDF of the normal distribution at these
% points. Then interpolation (pchip by default) is used to map intermediate
% points. An optional nonlinear transformation (sqrt or log) can be applied
% first. The transformation that is considered "best" is the one that results in
% the smallest max value of the 2nd derivative of the final transform.
% 
% x = unnormalized data
%
% xT = transformed data
%
% transform = function handle of the transform
% 
% Options:
% 'setPts': a 1D array specifying points where the CDF will be pinned to the
% normal distribution CDF. Some points may be removed if there is insufficient
% unique ranks in the data for uniqueness of these pinned points. Actual pinning
% points are based on tie adjusted rank of the data and ranks that are closest
% to the setPts are used.
%
% 'interpType': the type of interpolation to use between setPts. Default =
% 'pchip'.
%
% 'isPinEnds': boolean indicating whether to use min(x) and max(x) as additional
% setPts. Default = true.
%
% 'isTestTransforms': boolean indicating to test if sqrt or log transform
% results in "smoother" (more constant first derivative) final transformation.

p = inputParser();
p.addParamValue('setPts', [5, 25, 50, 75, 95]);
p.addParamValue('interpType', 'pchip');
p.addParamValue('isPinEnds', true);
p.addParamValue('isTestTransforms', true);
parse(p, varargin{:});


interpType = p.Results.interpType;
setPts = p.Results.setPts;
isPinEnds = p.Results.isPinEnds;
isTestTransforms = p.Results.isTestTransforms;
N = sum(~isnan(x));
xMin = min(x);

%% Compute pts for interpolant
cdfVals = (tiedrank(x) - .5)/N;
xPts = [];
cdfPts = [];

if isPinEnds % pin down lowest rank
   [cdfVal, cdfIdx] = min(cdfVals);
   xPts(1) = x(cdfIdx);
   cdfPts(1) = cdfVal;
end

for setPt = setPts
   [~, cdfIdx] = min(abs(cdfVals - setPt/100));
   if isempty(xPts) || x(cdfIdx) > xPts(end)
      xPts(end+1) = x(cdfIdx); %#ok<*AGROW>
      cdfPts(end+1) = cdfVals(cdfIdx);
   end
end

if isPinEnds % pin down largest rank
   [cdfVal, cdfIdx] = max(cdfVals);
   if x(cdfIdx) > xPts(end)
      xPts(end+1) = x(cdfIdx);
      cdfPts(end+1) = cdfVal;
   end
end
y = norminv(cdfPts);

%% Compute transformed data

if isTestTransforms % test to see if sqrt or log transform fits better
                    % based on approximation to total absolute curvature
   scores = nan(1, 3);
   scores(1) = sum(abs(diff(diff(y)./diff(xPts))));
   if xMin > 0
      scores(2) = sum(abs(diff(diff(y)./diff(sqrt(xPts)))));
      scores(3) = sum(abs(diff(diff(y)./diff(log(xPts)))));
   else
      scores(2) = sum(abs(diff(diff(y)./diff(sqrt(xPts - xMin + .25) - .5))));
      scores(3) = sum(abs(diff(diff(y)./diff(log(xPts - xMin + 1)))));
   end
   
   [~, scoreIdx] = min(scores);
   switch scoreIdx
      case 1
         transform = griddedInterpolant(xPts, y, interpType, interpType);
         xT = transform(x);
      case 2
         if xMin > 0
            interpolant = griddedInterpolant(sqrt(xPts), y, interpType, interpType);
            transform = @(x) interpolant(sqrt(x));
            xT = transform(x);
         else
            interpolant = griddedInterpolant(sqrt(xPts - xMin + .25) - .5, ...
                                             y, interpType, interpType);
            transform = @(x) interpolant(sqrt(x - xMin + .25) - .5);
            xT = transform(x);
         end
      case 3
         if xMin > 0
            interpolant = griddedInterpolant(log(xPts), y, interpType, interpType);
            transform = @(x) interpolant(log(x));
            xT = transform(x);
         else
            interpolant = griddedInterpolant(log(xPts - xMin + 1), y, interpType, ...
                                                                      interpType);
            transform = @(x) interpolant(log(x - xMin + 1));
            xT = transform(x);
         end
   end
else
   transform = griddedInterpolant(xPts, y, interpType, interpType);
   xT = transform(x);
end


end

