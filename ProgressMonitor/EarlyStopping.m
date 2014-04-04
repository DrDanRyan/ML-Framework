classdef EarlyStopping < BasicMonitor
   % Uses validation loss history to send stopping criteria flag. Will send stop
   % signal if best validation loss so far is too far in the past.
   
   properties
      % number of validation intervals to continue without improvement before 
      % sending stop signal (should be integer >= 0)
      bufferConst 
      
      % additional validation intervals based on percentage of intervals already
      % past; should be real >= 1
      bufferMult 
      
      % number of validation intervals before early stopping criteria is
      % checked; this acts like a minimum number of intervals before stopping
      burnIn
   end
   
   methods
      function obj = EarlyStopping(bufferConst, bufferMult, burnIn, varargin)
         obj = obj@BasicMonitor(varargin{:});
         obj.bufferConst = bufferConst;
         obj.bufferMult = bufferMult;
         obj.burnIn = burnIn;
      end
      
      function isContinue = should_continue(obj)
         isContinue = (obj.nUpdates < floor(obj.bufferConst + ...
                obj.bufferMult*obj.bestUpdate/obj.validationInterval)*...
                obj.validationInterval) | ...
                (obj.nUpdates < obj.burnIn*obj.validationInterval);
      end
   end
   
end

