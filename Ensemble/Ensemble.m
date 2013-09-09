classdef Ensemble < handle
   
   properties
      models
      preprocessors
      gpuState % should correspond to gpuState of model outputs
      outputSize % The length of the output from a single model
   end
   
   methods
      function obj = Ensemble(models, preprocessors, outputSize)
         obj.models = models;
         obj.preprocessors = preprocessors;
         obj.gpuState = models{1}.gpuState;
         if nargin < 3
            outputSize = 1;
         end
         obj.outputSize = outputSize;
         
      end
     
      function y = output(obj, x)
         ensembleSize = length(obj.models);
         y = obj.gpuState.zeros(obj.outputSize*ensembleSize, size(x, 2));
         for i = 1:ensembleSize
            startIdx = (i-1)*obj.outputSize + 1;
            stopIdx = i*obj.outputSize;
            y(startIdx:stopIdx, :) = obj.models{i}.output(obj.preprocessors{i}.transform(x));
         end
      end
      
      function gather(obj)
         obj.models = cellfun(@gather, obj.models, 'UniformOutput', false);
         obj.preprocessors = cellfun(@gather, obj.preprocessors, 'UniformOutput', false);
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.models = cellfun(@push_to_GPU, obj.models, 'UniformOutput', false);
         obj.preprocessors = cellfun(@push_to_GPU, obj.preprocessors, 'UniformOutput', false);
         obj.gpuState.isGPU = true;
      end
   end
   
end

