classdef EarlyStopping < BasicMonitor
   
   properties
      bufferConst % number of validation intervals to continue without improvement before quitting (should be integer >= 0)
      bufferMult  % should be real >= 1
   end
   
   methods
      function obj = EarlyStopping(bufferConst, bufferMult, varargin)
         obj = obj@BasicMonitor(varargin{:});
         obj.bufferConst = bufferConst;
         obj.bufferMult = bufferMult;
      end
      
      function isContinue = should_continue(obj)
         isContinue = obj.nUpdates < floor(obj.bufferConst + ...
                obj.bufferMult*obj.bestUpdate/obj.validationInterval)*obj.validationInterval;
      end
   end
   
end

