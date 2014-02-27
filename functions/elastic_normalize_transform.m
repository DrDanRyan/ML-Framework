function [xT, transform] = elastic_normalize_transform(x, varargin)

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

