classdef FullBatch < DataManager
   
   methods
      function obj = FullBatch(trainingInputs, trainingTargets, validationInputs, ...
                                       validationTargets)
         obj.trainingInputs = trainingInputs;
         obj.trainingTargets = trainingTargets;
         obj.validationInputs = validationInputs;
         obj.validationTargets = validationTargets;
      end
      
      function [x, t, endOfEpochFlag] = next_batch(obj)
         x = obj.trainingInputs;
         t = obj.trainingTargets;
         endOfEpochFlag = true;
      end
      
      function reset(obj)
         % pass
      end
   end
end

