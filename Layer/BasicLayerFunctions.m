classdef BasicLayerFunctions < handle
   
   properties (Abstract)
      params
      gpuState
   end
   
   methods
      function gather(obj)
         obj.params = cellfun(@(p) gather(p), obj.params, 'UniformOutput', false);
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.params = cellfun(@(p) single(gpuArray(p)), obj.params, 'UniformOutput', false);
         obj.gpuState.isGPU = true;
      end
      
      function increment_params(obj, delta)
         obj.params = cellfun(@plus, obj.params, delta, 'UniformOutput', false);
      end
   end
   
end

