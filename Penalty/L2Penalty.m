classdef L2Penalty < matlab.mixin.Copyable
   
   properties
      C
   end
   
   methods
      function obj = L2Penalty(C)
         obj.C = C;
      end
      
      function value = penalty_gradient(obj, layer)
         value = cellfun(@(p) obj.C*p, layer.params, 'UniformOutput', 'false');
      end
   end
   
end