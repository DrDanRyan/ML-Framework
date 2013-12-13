function [loss, gradVec] = SparseFilteringObj(paramVec, layer, data, paramShape)

gpuState = GPUState(isa(data, 'gpuArray'));
paramVecSize = size(paramVec);
if gpuState.isGPU
   paramVec = gpuArray(single(paramVec));
end

startIdx = 1;
for i = 1:length(paramShape)
   sizeVec = paramShape{i};
   stopIdx = startIdx + prod(sizevec) - 1;
   layer.params{i} = reshape(paramVec(startIdx:stopIdx), sizeVec);
end
clear paramVec

% Feed-forward
y = layer.feed_forward(data, true);
rowNorms = sqrt(sum(y.*y, 2));
yRowNormed = bsxfun(@rdivide, y, rowNorms);
colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
F = bsxfun(@rdivide, yRowNormed, colNorms);
loss = sum(abs(F(:)));

dLdy = gpuState.ones(size(F));
dLdy = bsxfun(@rdivide, dLdy, colNorms) ...
            - bsxfun(@times, F, sum(dLdy.*yRowNormed, 1)./(colNorms.*colNorms));
clear F colNorms         
dLdy = bsxfun(@rdivide, dLdy, rowNorms) ...
            - bsxfun(@times, yRowNormed, sum(dLdy.*y, 2)./(rowNorms.*rowNorms));
clear yRowNormed rowNorms

grad = layer.backprop(data, y, dLdy);
clear y dLdy

startIdx = 1;
gradVec = gpuState.nan(paramVecSize);
for i = 1:length(paramShape)
   stopIdx = startIdx + prod(sizevec) - 1;
   gradVec(startIdx:stopIdx) = reshape(grad{i}, [], 1);
end

if obj.gpuState.isGPU
   gradVec = gather(gradVec);
end

