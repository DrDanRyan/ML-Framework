function centers = myKMeans(data, nCenters, batchSize, maxIters)
[D, N] = size(data);
gpuState = GPUState(isa(data, 'gpuArray'));
centers = permute(data(:,randperm(N, nCenters)), [1, 3, 2]); % D x 1 x nCenters
delta = Inf;
iter = 0;

if nargin < 3
   batchSize = [];
else
   nBatches = ceil(N/batchSize);
end

if nargin < 4
   maxIters = 1e3;
end

while iter < maxIters && delta > 0
   
   % Compute distances to centroids
   if isempty(batchSize)
      differnces = bsxfun(@minus, data, centers); % D x N x nCenters
      distSquared = sum(differnces.*differnces,1); % 1 x N x nCenters
      clear differences
   else
      distSquared = gpuState.zeros(1, N, nCenters);
      startIdx = 1;
      for j = 1:nBatches
         stopIdx = min(N, startIdx + batchSize - 1);
         differences = bsxfun(@minus, data(:,startIdx:stopIdx), centers);
         distSquared(:,startIdx:stopIdx,:) = sum(differences.*differences, 1);
         startIdx = stopIdx + 1;
      end
      clear differences
   end
   
   % Compute membership vector
   membership = bsxfun(@eq, distSquared, min(distSquared, [], 3)); % 1 x N x nCenters
   clear distSquared
   
   % Compute new centers
   new_centers = gpuArray.zeros(D, 1, nCenters);
   for k = 1:nCenters
      members = membership(:,:,k);
      new_centers(:,1,k) = sum(data(:,members), 2)/sum(members);
   end
   delta = max(abs(new_centers(:) - centers(:)));
   clear membership 
   centers = new_centers;
   clear new_centers;
   
   iter = iter + 1;
   fprintf('iter: %d   delta: %d\n', iter, delta);
end

centers = squeeze(centers);
end