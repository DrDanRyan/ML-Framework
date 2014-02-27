function [xT, transform] = elastic_normalize_transform(x, varargin)
% setPts is a list of percentiles that will be exactly pinned to the
% corresponding point in the normal distribution. All points between the
% setPts are linearly interpolated.

p = inputParser();
p.addParamValue('setPts', [5, 25, 50, 75, 95]);
p.addParamValue('interpType', 'pchip');
p.addParamValue('isPinEnds', true);
parse(p, varargin{:});

interpType = p.Results.interpType;
setPts = p.Results.setPts;
isPinEnds = p.Results.isPinEnds;
N = sum(~isnan(x));

if isPinEnds
   xP = prctile(x, [0, setPts, 100]);
   setPts = [50/N, setPts, 100-50/N];
else
   xP = prctile(x, setPts);
end

if any(diff(xP) == 0)
   ranks = tiedrank(x);
   while any(diff(xP) == 0) % Need to remove repeated values, replace with tiedrank prctile
      i = find(diff(xP)==0, 1);
      xVal = xP(i);
      xIdx = find(x==xVal, 1);
      xRank = ranks(xIdx);
      xPrct = (xRank - .5)*100/N;
      xP = xP([1:i, i+2:end]);
      setPts = [setPts(1:i-1), xPrct, setPts(i+2:end)];
   end
end

y = norminv(setPts/100);
transform = griddedInterpolant(xP, y, interpType, interpType);
xT = transform(x);
end

