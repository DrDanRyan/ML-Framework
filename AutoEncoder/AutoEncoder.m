classdef AutoEncoder < handle
   % Generic denoising AutoEncoder
   
   properties
      encodeLayer % a HiddenLayer object that functions as the encoding layer
      decodeLayer % a OutputLayer object that functions as the decoding layer and loss function
      isTiedWeights % a boolean indicating if the params in encodeLayer and decodeLayer are shared
      inputDropout
      hiddenDropout
      gpuState
   end
   
   methods
      function obj = AutoEncoder(varargin)
         p = inputParser;
         p.addParamValue('inputDropout', 0);
         p.addParamValue('hiddenDropout', 0);
         p.addParamValue('encodeLayer', []);
         p.addParamValue('decodeLayer', []);
         p.addParamValue('isTiedWeights', false);
         p.addParamValue('gpu', []);
         parse(p, varargin{:});
         
         obj.encodeLayer = p.Results.encodeLayer;
         obj.decodeLayer = p.Results.decodeLayer;
         obj.inputDropout = p.Results.inputDropout;
         obj.hiddenDropout = p.Results.hiddenDropout;
         obj.isTiedWeights = p.Resutls.isTiedWeights;
         obj.gpuState = GPUState(p.Results.gpu);
      end
      
      function [grad, xRecon] = gradient(obj, x)
         xCorrupt = x.*obj.gpuState.binary_mask(size(x), obj.inputDropout);
         xCode = obj.encodeLayer.feed_forward(xCorrupt);
         xCode = xCode.*obj.gpuState.binary_mask(size(xCode), obj.hiddenDropout);
         [decodeGrad, dLdxCode, xRecon] = obj.decodeLayer.backprop(xCode, x);
         encodeGrad = obj.encodeLayer.backprop(xCorrupt, dLdxCode);
         
         if obj.isTiedWeights
            grad = cellfun(@(g1, g2) g1 + g2', encodeGrad, decodeGrad, ...
                              'UniformOutput', false);
         else
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function loss = compute_loss(obj, xRecon, x)
         loss = obj.decodeLayer.compute_loss(xRecon, x);
      end
      
      function xRecon = output(obj, x)
         x = (1-obj.inputDropout).*x;
         xCode = (1-obj.hiddenDropout).*obj.encodeLayer.feed_forward(x);
         xRecon = obj.decodeLayer.feed_forward(xCode);
      end
      
      function update_params(obj, delta_params)
         if obj.isTiedWeights
            obj.encodeLayer.increment_params(delta_params);
            obj.decodeLayer.params = obj.get_encode_params_transposed();
         else
            splitIdx = length(obj.encodeLayer.params);
            obj.encodeLayer.increment_params(delta_params(1:splitIdx));
            obj.decodeLayer.increment_params(delta_params(splitIdx+1:end));
         end
      end
      
      function gather(obj)
         obj.encodeLayer.gather();
         obj.decodeLayer.gather();
      end
      
      function push_to_GPU(obj)
         obj.encodeLayer.push_to_GPU();
         obj.decodeLayer.push_to_GPU();
      end
      
      function reset(obj)
         obj.encodeLayer.reset();
         if obj.isTiedWeights
            obj.decodeLayer.params = obj.get_encode_params_transposed();
         else
            obj.decodeLayer.reset();
         end
      end
      
      function pTrans = get_encode_params_transposed(obj)
         pTrans = cellfun(@(p) p', obj.encodeLayer.params, 'UniformOutput', false);
      end
   end
   
end

