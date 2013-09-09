classdef ConsoleReporter < Reporter
   
   properties
      epoch = 0;
   end
   
   methods
      function update(obj, trainingLoss, validationLoss)
         obj.epoch = obj.epoch + 1;
         fprintf('Epoch %d: \t train: %d \t valid: %d \n', obj.epoch, trainingLoss, validationLoss);
      end
      
      function reset(obj)
         obj.epoch = 0;
      end
   end
   
end

