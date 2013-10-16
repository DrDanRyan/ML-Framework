classdef BasicMomentumSchedule < TrainingSchedule
   % Fixed learning rate and momentum. Fixed number of epochs.
   
   properties
      params % {learnRate, momentum}
      maxEpochs % number of epochs to train before terminating
      epoch = 0;
   end
   
   methods
      function obj = BasicMomentumSchedule(maxEpochs, learnRate, momentum)
         obj.params{1} = learnRate;
         obj.params{2} = momentum;
         obj.maxEpochs = maxEpochs;
      end
      
      function isContinue = update(obj, ~, ~, ~)
         % Increment epoch counter and check if maximum number of epochs
         % has been reached.
         obj.epoch = obj.epoch + 1;
         isContinue = obj.epoch < obj.maxEpochs;
      end
      
      function reset(obj)
         % Reset epoch counter to 0.
         obj.epoch = 0;
      end
   end
   
end

