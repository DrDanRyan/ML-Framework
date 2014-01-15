classdef RandomProjection < Preprocessor
   % Projects data onto a random lower dimensional subspace using a
   % Gaussian distribution to generate random vectors that are then
   % normalized to unit length (in L2 norm). Resulting vectors are NOT
   % orthogonal.
   
   properties
      inputDim
      outputDim
      gpuState
      P % projection matrix
   end
   
   methods
      function obj = RandomProjection(inputDim, outputDim, gpu)
         if nargin < 3
            gpu = [];
         end
         obj.inputDim = inputDim;
         obj.outputDim = outputDim;
         obj.gpuState = GPUState(gpu);
         obj.init_params();
      end
      
      function x = feed_forward(obj, x)
         x = obj.P*x;
      end
      
      function gather(obj)
         obj.P = gather(obj.P);
      end
      
      function push_to_GPU(obj)
         obj.P = single(gpuArray(obj.P));
      end
      
      function init_params(obj)
         obj.P = obj.gpuState.randn(obj.outputDim, obj.inputDim);
         columnNorms = sqrt(sum(obj.P.^2, 1));
         obj.P = bsxfun(@rdivide, obj.P, columnNorms);
      end
   end
   
end

