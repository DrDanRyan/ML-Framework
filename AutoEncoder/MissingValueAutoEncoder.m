classdef MissingValueAutoEncoder < AutoEncoder
   % This AutoEncoder variant accepts input data with NaN fields. The NaN
   % values are replaced with zero for the encoding stage and the reconstruction
   % of these fields does not contribute to the loss.
   
   methods
      function obj = MissingValueAutoEncoder(varargin)
         obj = obj@AutoEncoder(varargin{:});
      end
      
      function [grad, xRecon] = gradient(obj, x, ~, ~)
         isNaN = isnan(x);
         x(isNaN) = 0;
         xCorrupt = x.*obj.gpuState.binary_mask(size(x), obj.inputDropout);
         xCode = obj.encodeLayer.feed_forward(xCorrupt);
         xCode = xCode.*obj.gpuState.binary_mask(size(xCode), obj.hiddenDropout);

         [dLdz, xRecon] = obj.decodeLayer.dLdz(xCode, x);
         dLdz(isNaN) = 0;
         [decodeGrad, dLdxCode] = obj.decodeLayer.backprop(xCode, [], true, dLdz);
         encodeGrad = obj.encodeLayer.backprop(xCorrupt, xCode, dLdxCode);
         
         if obj.isTiedWeights
            grad = cellfun(@(g1, g2) g1 + g2', encodeGrad, decodeGrad, ...
                              'UniformOutput', false);
         else
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function loss = compute_loss(obj, xRecon, x)
         isNaN = isnan(x);
         x(isNaN) = xRecon(isNaN);
         loss = obj.decodeLayer.compute_loss(xRecon, x);
      end
      
      function xCode = encode(obj, x)
         x(isnan(x)) = 0;
         xCode = (1-obj.inputDropout)*obj.encodeLayer.feed_forward(x);
      end
      
      function xRecon = output(obj, x)
         x(isnan(x)) = 0;
         x = (1-obj.inputDropout).*x;
         xCode = (1-obj.hiddenDropout).*obj.encodeLayer.feed_forward(x);
         xRecon = obj.decodeLayer.feed_forward(xCode);
      end
   end
   
end

