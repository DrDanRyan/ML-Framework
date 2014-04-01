classdef WeightDecayPenalty < handle
   % A mixin that provides L1 and L2 weight decay penalties. 
   
   properties (Abstract)
      params
   end
   
   properties (Dependent = true, SetAccess = private)
      isWeightDecay % a boolean flag indicating if any penalty is on
   end
   
   properties
      L1Penalty % a positive coefficient for L1 weight decay
      L2Penalty % a positive coefficient for L2 weight decay
   end
   
   methods
      function obj = WeightDecayPenalty(varargin)       
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('L1Penalty', 0, @(x) x >= 0);
         p.addParamValue('L2Penalty', 0, @(x) x >= 0);
         parse(p, varargin{:});
         
         obj.L1Penalty = p.Results.L1Penalty;
         obj.L2Penalty = p.Results.L2Penalty;         
      end
      
      function flag = get.isWeightDecay(obj)
         if (obj.L1Penalty==0 && obj.L2Penalty==0)
            flag = false;
         else
            flag = true;
         end
      end        
      
      function penalty = compute_weight_decay_penalty(obj)
         penalty = obj.L1Penalty*sign(obj.params{1}) + ...
                   2*obj.L2Penalty*obj.params{1};
      end
   end
   
end

