classdef MiniBatch < DataManager
   
   properties
      batchSize
      trainingSize
      startIdx
      stopIdx
   end
   
   methods
      function obj = BasicDataManager(batchSize, trainingInputs, trainingTargets, validationInputs, ...
                                       validationTargets)
         obj.batchSize = batchSize;
         
         if nargin > 1
            obj.trainingInputs = trainingInputs;
            obj.trainingTargets = trainingTargets;
            obj.validationInputs = validationInputs;
            obj.validationTargets = validationTargets;
            obj.trainingSize = size(trainingInputs, 2);
            obj.shuffle_training_data();
         end
      end
      
      function [x, t, endOfEpochFlag] = next_batch(obj)
         x = obj.trainingInputs(:, obj.startIdx:obj.stopIdx);
         t = obj.trainingTargets(:, obj.startIdx:obj.stopIdx);
         
         if obj.stopIdx == obj.trainingSize
            endOfEpochFlag = true;
            obj.shuffle_training_data();
         else
            endOfEpochFlag = false;
            obj.startIdx = obj.startIdx + obj.batchSize;
            obj.stopIdx = min(obj.trainingSize, obj.stopIdx + obj.batchSize);
         end
      end
      
      function shuffle_training_data(obj)
         [obj.trainingInputs, obj.trainingTargets] = shuffle(obj.trainingInputs, ...
                                                             obj.trainingTargets);
         obj.startIdx = 1;
         obj.stopIdx = min(obj.batchSize, obj.trainingSize);
      end
      
      function reset(obj)
         obj.trainingSize = size(obj.trainingInputs, 2);
         obj.shuffle_training_data();
      end
   end
   
end

