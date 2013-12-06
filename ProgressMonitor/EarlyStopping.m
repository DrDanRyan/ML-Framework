classdef EarlyStopping < BasicMonitor
   
   properties
      bufferConst % number of validation intervals to continue without improvement before quitting (should be integer >= 0)
      bufferMult  % should be real >= 1
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
                obj.bufferMult*obj.bestUpdate/obj.validationInterval)*obj.validationInterval) | ...
                (obj.nUpdates < obj.burnIn*obj.validationInterval);
      end
   end
   
end

