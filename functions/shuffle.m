function [shuffledArray1, shuffledArray2] = shuffle(array1, array2)
% Takes two arrays with the same number of columns and randomly permutes
% the columns in an identical manner.  Used to shuffle data between epochs
% (assuming that each COLUMN corresponds to an example).

if size(array1, 2) ~= size(array2, 2)
   exception = MException('VerifyInput:DimensionMismatch', ...
      'Input arrays must have same number of columns.');
   throw(exception);
end

len = size(array1, 2);
permVec = randperm(len);

shuffledArray1 = array1(:, permVec);
shuffledArray2 = array2(:, permVec);
end

