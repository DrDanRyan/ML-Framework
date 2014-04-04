classdef Reporter < matlab.mixin.Copyable
   % This defines the Reporter interface
   
   methods (Abstract)
      report(obj, progressMonitor, model)
      reset(obj)
   end
   
end

