classdef ParameterSchedule < matlab.mixin.Copyable
   % This defines the ParameterSchedule interface
   
   methods (Abstract)
      params = update(obj)
      reset(obj)
   end
   
end

