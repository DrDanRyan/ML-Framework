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
   
   methods
      function obj = DataManager(trainingInputs, trainingTargets, validationInputs, ...
                                       validationTargets, varargin)
         
         if nargin > 0
            obj.trainingInputs = trainingInputs;
            obj.trainingTargets = trainingTargets;
            obj.validationInputs = validationInputs;
            obj.validationTargets = validationTargets;
         end
         
         p = inputParser();
         p.addParamValue('batchSize', []);
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
      end
      
      function [x, t, endOfEpochFlag] = next_batch(obj)
         if isempty(obj.batchSize) % full batch
            x = obj.trainingInputs;
            t = obj.trainingTargets;
            endOfEpochFlag = true;
         else % mini-batch
            if isempty(obj.trainingInputs)
               x = [];
            else
               x = obj.trainingInputs(:, obj.startIdx:obj.stopIdx);
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