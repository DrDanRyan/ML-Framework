classdef ParamsLayer < handle
   % A class to mixin with HiddenLayer or OutputLayer for layers that have parameters
  
   properties
      initType
      initScale
      gpuState
      params
   end
   
   methods
      function obj = ParamsLayer(varargin)
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('gpu', []);
         p.addParamValue('initType', 'dense');
         p.addParamValue('initScale', .005);
         parse(p, varargin{:});
         obj.gpuState = GPUState(p.Results.gpu);
         obj.initType = p.Results.initType;
         obj.initScale = p.Results.initScale;
      end
      
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

