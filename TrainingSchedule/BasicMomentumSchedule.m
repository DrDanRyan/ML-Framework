classdef BasicMomentumSchedule < TrainingSchedule
   % Fixed learnRate and momentum. Fixed number of epochs.
   
   properties
      params % {learnRate, momentum}
      maxEpochs
      epoch = 0;
   end
   
   methods
      function obj = BasicMomentumSchedule(learnRate, momentum, maxEpochs)
         obj.params{1} = learnRate;
         obj.params{2} = momentum;
         obj.maxEpochs = maxEpochs;
      end
      
      function isContinue = update(obj, ~, ~, ~)
         obj.epoch = obj.epoch + 1;
         isContinue = obj.epoch < obj.maxEpochs;
      end
      
      function reset(obj)
         obj.epoch = 0;
      end
   end
   
end

