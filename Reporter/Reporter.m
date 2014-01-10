classdef Reporter < matlab.mixin.Copyable
   % Defines the Reporter interface
   
   methods (Abstract)
      report(obj, progressMonitor, model)
      reset(obj)
   end
   
end

