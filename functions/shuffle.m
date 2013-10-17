function [shuffledArray1, shuffledArray2] = shuffle(array1, array2)
% Takes two arrays with the same number of columns and randomly permutes
% the columns in an identical manner.  Used to shuffle data between epochs
% (assuming that each COLUMN corresponds to an example).

if ~isempty(array1) && ~isempty(array2) && size(array1, 2) ~= size(array2, 2)
   exception = MException('VerifyInput:DimensionMismatch', ...
      'Input arrays must have same number of columns.');
   throw(exception);
end

if isempty(array1)
   len = size(array2, 2);
else
   len = size(array1, 2);
end
permVec = randperm(len);

if isempty(array1)
   shuffledArray1 = [];
else
   shuffledArray1 = array1(:, permVec);
end

if isempty(array2)
   shuffledArray2 = [];
else
   shuffledArray2 = array2(:, permVec);
end
end

