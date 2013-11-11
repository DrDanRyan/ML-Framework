classdef ProgressMonitor < matlab.mixin.Copyable
   % This defines the ProgressMonitor interface.
   
   methods (Abstract)
      isContinue = update(obj, model, dataManager)
      reset(obj)
   end
   
end

