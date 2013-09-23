function hold_outs = stratified_cross_validate_partition(targets, nFolds)
% Creata a cross-validation partition where the ratio of positive to
% negative examples is roughly equal in each fold. Assumes that targets is
% either a binary vector (zeros and ones) or a vector of +1 and -1. Hold
% outs should be shuffled before being fed to training algorithm.

% Determine number of targets and create idxs vector
N = length(targets);
idxs = 1:N;

% Sepearate indexes that correspond to positive and negative examples
positives = idxs(targets == 1);
negatives = idxs(targets ~= 1);

% Compute approximate foldSize for positive and negative examples
nPos = length(positives);
nNeg = N - nPos;
posFoldSize = floor(nPos/nFolds);
negFoldSize = floor(nNeg/nFolds);

% Shuffle positive and negative indexes
positives = positives(randperm(nPos));
negatives = negatives(randperm(nNeg));

hold_outs = cell(1, nFolds);
posStart = 1;
posStop = posFoldSize;
negStart = 1;
negStop = negFoldSize;
for i = 1:nFolds
   idxs = [positives(posStart:posStop), negatives(negStart:negStop)];
   posStart = posStop + 1;
   posStop = posStart + posFoldSize -1;
   negStart = negStop + 1;
   negStop = negStart + negFoldSize - 1;
   % Add remainder positive examples
   if i <= rem(nPos, nFolds)
      idxs = [idxs, positives(nFolds*posFoldSize + i)];
   end
   
   % Add remainder negative examples
   if i <- rem(nNeg, nFolds)
      idxs = [idxs, negatives(nFolds*negFoldSize + i)];
   end
   hold_outs{i} = idxs;
end

end

