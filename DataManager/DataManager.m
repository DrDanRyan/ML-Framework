classdef DataManager < matlab.mixin.Copyable
   
   properties
      trainingInputs
      trainingTargets
      validationInputs
      validationTargets
      
      batchSize
      trainingSize
      startIdx
      stopIdx
   end
   
   properties (Dependent)
      isGPU
   end
   
   methods
      function obj = DataManager(trainingInputs, trainingTargets, validationInputs, ...
                                       validationTargets, varargin)

         obj.trainingInputs = trainingInputs;
         obj.trainingTargets = trainingTargets;
         obj.validationInputs = validationInputs;
         obj.validationTargets = validationTargets;
         
         p = inputParser();
         p.addParamValue('batchSize', []);
         p.addParamValue('noise', []);
         parse(p, varargin{:});
         
         if ~isempty(p.Results.batchSize)
            p.batchSize = p.Results.batchSize;
            if ~isempty(obj.trainingInputs)
               obj.trainingSize = size(obj.trainingInputs, 2);
            elseif ~isempty(obj.trainingTargets)
               obj.trainingSize = size(obj.trainingInputs, 2);
            end
            obj.shuffle_training_data();
         end
         
         if ~isempty(p.Results.noise)
            obj.noiseLevel = p.Results.noise;
         end
      end
      
      
      
      function [x, t, endOfEpochFlag] = next_batch(obj)
         if isempty(obj.batchSize) % full batch
            x = obj.inject_noise(obj.trainingInputs);
            t = obj.trainingTargets;
            endOfEpochFlag = true;
         else % mini-batch
            if isempty(obj.trainingInputs)
               x = [];
            else
               x = obj.inject_noise(obj.trainingInputs(:, obj.startIdx:obj.stopIdx));
            end
            
            if isempty(obj.trainingTargets)
               t = [];
            else
               t = obj.trainingTargets(:, obj.startIdx:obj.stopIdx);
            end
            
            if obj.stopIdx == obj.trainingSize
               endOfEpochFlag = true;
               obj.shuffle_training_data();
            else
               endOfEpochFlag = false;
               obj.startIdx = obj.startIdx + obj.batchSize;
               obj.stopIdx = min(obj.trainingSize, obj.stopIdx + obj.batchSize);
            end
         end
      end
      
      function shuffle_training_data(obj)
         [obj.trainingInputs, obj.trainingTargets] = shuffle(obj.trainingInputs, ...
                                                             obj.trainingTargets);
         obj.startIdx = 1;
         obj.stopIdx = min(obj.batchSize, obj.trainingSize);
      end
      
      function isGPU = get.isGPU(obj)
         isGPU = isa(obj.trainingInputs, 'gpuArray');
      end
      
      function x = inject_noise(obj, x)
         if isempty(obj.noiseLevel)
            return
         end
         
         if obj.isGPU
            x = x + obj.noiseLevel*gpuArray.randn(size(x), 'single');
         else
            x = x + obj.noiseLevel*randn(size(x));
         end
      end
      
      function reset(obj)
         if ~isempty(obj.batchSize)
            if ~isempty(obj.trainingInputs)
               obj.trainingSize = size(obj.trainingInputs, 2);
            elseif ~isempty(obj.trainingTargets)
               obj.trainingSize = size(obj.trainingInputs, 2);
            end
            obj.shuffle_training_data();
         end
      end
   end
end