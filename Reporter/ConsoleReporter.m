classdef ConsoleReporter < Reporter
   
   methods
      function report(~, progressMonitor, ~)
         if isempty(progressMonitor.trainLoss)
            fprintf('Update %d:  valid: %7.5g\n', ...
                     progressMonitor.nUpdates, ...
                     progressMonitor.validLoss(end));
         else
            fprintf('Update %d:  train: %7.5g   valid: %7.5g\n', ...
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

