classdef MaxFanInConstraint < handle
   % A mixin that provides a L2 norm max fan-in constraint on output units.

   properties (Abstract)
      params
   end
   
   properties (Dependent = true, SetAccess = private)
      isMaxFanIn
   end
   
   properties
      % Maximum L2 norm allowed for fan-in to each output unit (row of weight
      % matrix). If this max value is exceeded after an update, the row is
      % rescaled to have this max value for its L2 norm.
      maxFanIn
   end
   
   methods
      function obj = MaxFanInConstraint(varargin)
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('maxFanIn', []);
         parse(p, varargin);
         
         obj.maxFanIn = p.Results.maxFanIn;
      end
      
      function flag = get.isMaxFanIn(obj)
         flag = ~isempty(obj.maxFanIn);
      end
      
      function impose_fanin_constraint(obj)
         rowNorms = sqrt(sum(obj.params{1}.^2, 2));
         multiplier = min(1, obj.maxFanIn./rowNorms);
         obj.params{1} = bsxfun(@times, obj.params{1}, multiplier);
      end
   end

end

