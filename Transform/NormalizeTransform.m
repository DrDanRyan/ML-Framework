classdef NormalizeTransform < matlab.mixin.Copyable

   properties
      % xTrans = a*x + b + c*sqrt(x + d) + e*log(x + f) where a, c, e >= 0,
      % also require d >= -min(x) and f >= -min(x) + 1
      params % = {a, b, c, d, e, f}
      xMin % = min(x)
      gpuState
   end
   
   methods
      function obj = NormalizeTransform(mu, sigma, xMin, gpu)
         obj.xMin = xMin;
         obj.params = {1/sigma, -mu/sigma, 0, max(0, -xMin), 0, max(0, -xMin + 1)};
         if nargin < 4
            obj.gpuState = GPUState(false);
         else
            obj.gpuState = GPUState(gpu);
         end
      end
      
      function xT = transform(obj, x)
         [a, b, c, d, e, f] = obj.params{:};
         xT = a*x + b + c*sqrt(x + d) + e*log(x + f);
      end
      
      function value = compute_dxTdp(obj, x)
         [~, ~, c, d, e, f] = obj.params{:};
         value = {x, 1, sqrt(x + d), c./(2*sqrt(x + d)), log(x + f), e./(x + f)};
      end
      
      function [grad, loss] = gradient(obj, batch)
         [x, y] = batch{:};
         xT = obj.transform(x);
         normcdf_xT = normcdf(xT);
         [loss, idx] = max(abs(normcdf_xT - y));
         dxTdp = obj.compute_dxTdp(x(idx));
         temp = sign(normcdf_xT(idx) - y(idx))*normpdf(xT(idx));
         grad = cell(1, 6);
         for i = 1:6
            grad{i} = temp*dxTdp{i};
         end
      end
      
      function loss = compute_loss_from_output(~, loss, ~)
         % pass
         % dummy function implemented to use IRprop
      end
      
      function increment_params(obj, delta)
         obj.params = cellfun(@plus, obj.params, delta, 'UniformOutput', false);
         % enforce a, c, e >= 0
         obj.params{1} = max(0, obj.params{1});
         obj.params{3} = max(0, obj.params{3});
         obj.params{5} = max(0, obj.params{5});
         
         % enforce d >= -xMin
         obj.params{4} = max(-obj.xMin, obj.params{4});
         
         % enforce f >= -xMin + 1
         obj.params{6} = max(-obj.xMin + 1, obj.params{6}); 
      end
   end
end
