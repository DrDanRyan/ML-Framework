classdef ConsoleReporter < Reporter
   
   methods
      function report(~, progressMonitor, ~)
         if isempty(progressMonitor.trainLoss)
            fprintf('Epoch %d:  valid: %.5g\n', ...
                     progressMonitor.nUpdates, ...
                     progressMonitor.validLoss(end));
         else
            fprintf('Epoch %d:  train: %.5g   valid: %.5g\n', ...
                     progressMonitor.nUpdates, ...
                     progressMonitor.trainLoss(end), ...
                     progressMonitor.validLoss(end));
         end
      end
      
      function reset(~)
         % pass
      end
      
   end
end

