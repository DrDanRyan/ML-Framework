classdef ConsoleReporter < Reporter
   % Basic text based console reporter that prints training loss and validation
   % loss to screen. It is common to add this reporter as an extra property to a
   % more advanced reporter if desired.
   
   methods
      function report(~, progressMonitor, ~)
         fprintf('\nUpdate %d:  ', progressMonitor.nUpdates);
         
         if ~isempty(progressMonitor.trainLoss)
            fprintf('train: %7.5g  ', progressMonitor.trainLoss(end));
         end
         
         if ~isempty(progressMonitor.validLoss)
            fprintf('valid: %7.5g', progressMonitor.validLoss(end));
         end
      end
      
      function reset(~)
         % pass
      end
      
   end
end

