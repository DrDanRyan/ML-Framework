classdef RegularizationFunctions < handle
   % A mixin that provides L1 and L2 weight penalties and a 
   % L2 max fan-in constraint on the output units (sqrt of sum of squared across 2nd dim)
   
   properties (Abstract)
      params
   end
   
   properties
      isPenalty
      L1Penalty
      L2Penalty
      maxFanIn
   end
   
   methods
      function obj = RegularizationFunctions(varargin)       
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('L1Penalty', []);
         p.addParamValue('L2Penalty', []);
         p.addParamValue('maxFanIn', []);
         parse(p, varargin{:});
         
         obj.maxFanIn = p.Results.maxFanIn;
         obj.L1Penalty = p.Results.L1Penalty;
         obj.L2Penalty = p.Results.L2Penalty;
         if (isempty(obj.L1Penalty) && isempty(obj.L2Penalty))
            obj.isPenalty = false;
         else
            obj.isPenalty = true;
         end
      end
      
      function penalties = compute_penalties(obj)
         if isempty(obj.L1Penalty) % L2Penalty only
            penalties = cellfun(@(p) obj.L2Penalty*p, obj.params, 'UniformOutput', false);
         elseif isempty(obj.L2Penalty) % L1Penalty only
            penalties = cellfun(@(p) obj.L1Penalty*sign(p), obj.params, 'UniformOutput', false); 
         else % Both L1 and L2 penalties are active
            penalties = cellfun(@(p) obj.L1Penalty*sign(p) + obj.L2Penalty*p, obj.params, ...
                                    'UniformOutput', false);
         end
      end
      
      function impose_fanin_constraint(obj)
         rowNorms = sqrt(sum(obj.params{1}.^2, 2));
         multiplier = min(1, obj.maxFanIn./rowNorms);
         obj.params{1} = bsxfun(@times, obj.params{1}, multiplier);
      end
   end
end

