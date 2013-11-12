function error_rate = compute_error_rate(y, t)
[~, yIdx] = max(y);
[~, tIdx] = max(t);
N = size(y, 2);
error_rate = sum(yIdx ~= tIdx)/N;
end

