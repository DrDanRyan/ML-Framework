function sample = multinomial_sample(probArray, dim)
   if nargin < 2
      dim = 1;
   end
   
   sampleSize = size(probArray);
   sampleSize(dim) = 1;
   isGPU = isa(probArray, 'gpuArray');

   if isGPU
    randProb = gpuArray.rand(sampleSize, 'single');
    padding = gpuArray.ones(sampleSize,'single');
   else
    randProb = rand(sampleSize);
    padding = ones(sampleSize);
   end

   probArray = cumsum(probArray, dim);
   if isGPU
    isBelowRand = cat(dim, padding, single(bsxfun(@lt, probArray, randProb)));
   else
    isBelowRand = cat(dim, padding, bsxfun(@lt, probArray, randProb));
   end
   sample = -diff(isBelowRand, 1, dim);
end

