classdef L1Penalty < matlab.mixin.Copyable
   
   properties
      C
   end
   
   methods
      function obj = L1Penalty(C)
         obj.C = C;
      end
      
      function value = penalty_gradient(obj, layer)
         value = cellfun(@(p) obj.C*sign(p), layer.params, 'UniformOutput', 'false');
      end
   end
   
end

