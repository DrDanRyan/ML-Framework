function [loss, gradVec] = SparseFilteringObj(paramVec, layer, data, paramShape, batchSize)

N = size(data, 2);
if nargin < 5
   batchSize = N;
end
gpuState = GPUState(isa(data, 'gpuArray'));
paramVecSize = size(paramVec);
if gpuState.isGPU
   paramVec = gpuArray(single(paramVec));
end

startIdx = 1;
for i = 1:length(paramShape)
   sizeVec = paramShape{i};
   stopIdx = startIdx + prod(sizeVec) - 1;
   layer.params{i} = reshape(paramVec(startIdx:stopIdx), sizeVec);
   startIdx = stopIdx + 1;
end
clear paramVec

startIdx = 1;
grad = {};
while startIdx <= N
   % Get batch
   stopIdx = min(N, startIdx + batchSize - 1);
   batch = data(:, startIdx:stopIdx, :);
   batchSize = stopIdx - startIdx + 1;
   
   % Feed-forward
   y = layer.feed_forward(batch, true);
   rowNorms = sqrt(sum(y.*y, 2));
   yRowNormed = bsxfun(@rdivide, y, rowNorms);
   colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
   F = bsxfun(@rdivide, yRowNormed, colNorms);
   loss = gather(sum(abs(F(:))));

   % Backprop
   dLdy = batchSize*gpuState.ones(size(F)); % multiply by batchSize to cancel averaging effect
   dLdy = bsxfun(@rdivide, dLdy, colNorms) ...
               - bsxfun(@times, F, sum(dLdy.*yRowNormed, 1)./(colNorms.*colNorms));
   clear F colNorms         
   dLdy = bsxfun(@rdivide, dLdy, rowNorms) ...
               - bsxfun(@times, yRowNormed, sum(dLdy.*y, 2)./(rowNorms.*rowNorms));
   clear yRowNormed rowNorms

   if isempty(grad)
      grad = layer.backprop(batch, y, dLdy);
   else
      grad = cellfun(@plus, grad, layer.backprop(batch, y, dLdy), 'UniformOutput', false);
   end
   clear y dLdy
   startIdx = stopIdx + 1;
end


% Reshape the gradient (and gather if it was on GPU)
startIdx = 1;
gradVec = gpuState.nan(paramVecSize);
for i = 1:length(paramShape)
   stopIdx = startIdx + numel(grad{i}) - 1;
   gradVec(startIdx:stopIdx) = reshape(grad{i}, [], 1);
   startIdx = stopIdx + 1;
end

if gpuState.isGPU
   gradVec = gather(gradVec);
end

