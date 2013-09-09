function hold_outs = cross_validate_partition(n, k)

hold_outs = cell(1, k);
permvec = randperm(n);
foldSize = floor(n/k);

for i = 1:k
   startIdx = (i-1)*foldSize + 1;
   endIdx = i*foldSize;
   idxs = permvec(startIdx:endIdx);
   if i <= rem(n, k)
      idxs = [idxs, permvec(k*foldSize + i)];
   end
   hold_outs{i} = idxs;
end

