function hold_outs = CV_partition(n, k)
% Computes k-fold cross-validation partition.
% 
% hold_outs is a 1 x k cell array. Each element is a list of indicies in that
% fold. 

hold_outs = cell(1, k);
permvec = randperm(n);
foldSize = floor(n/k);

for i = 1:k
   startIdx = (i-1)*foldSize + 1;
   endIdx = i*foldSize;
   idxs = permvec(startIdx:endIdx);
   if i <= rem(n, k)
      idxs = [idxs, permvec(k*foldSize + i)]; %#ok<AGROW>
   end
   hold_outs{i} = idxs;
end

