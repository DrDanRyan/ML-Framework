classdef AutoEncoder < AutoEncoderInterface
   % Basic AutoEncoder
   
   properties
      encodeLayer % a HiddenLayer object that functions as the encoding layer
      decodeLayer % a OutputLayer object that functions as the decoding layer 
      
      % a boolean indicating if the params in encodeLayer and decodeLayer are 
      % shared; assumes encodeLayer.params = {W, b} when isTiedWeights is true
      isTiedWeights 
      
      gpuState
      encodeGradSize % size of cell array returned by enocdeLayer.backprop
   end
   
   methods
      function obj = AutoEncoder(varargin)
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('isTiedWeights', false);
         p.addParamValue('gpu', []);
         parse(p, varargin{:});

         obj.isTiedWeights = p.Results.isTiedWeights;
         obj.gpuState = GPUState(p.Results.gpu);
      end
      
      function set.decodeLayer(obj, decodeLayer)
         % Ties the weights of the decodeLayer to the encodeLayer if
         % isTiedWeights is true.
         obj.decodeLayer = decodeLayer;
         if obj.isTiedWeights %#ok<MCSUP>
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         end
      end
      
      function [grad, xRecon] = gradient(obj, batch)
         x = batch{1};
         h = obj.encodeLayer.feed_forward(x, true);
         [decodeGrad, dLdh, xRecon] = obj.decodeLayer.backprop(h, x);
         encodeGrad = obj.encodeLayer.backprop(x, h, dLdh);
         obj.encodeGradSize = length(encodeGrad);
         
         if obj.isTiedWeights
            grad = obj.tied_weights_grad(encodeGrad, decodeGrad);
         else
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function grad = tied_weights_grad(~, encodeGrad, decodeGrad)
         if ndims(encodeGrad{1}) <= 2
            grad{1} = encodeGrad{1}+decodeGrad{1}';
         else
            grad{1} = encodeGrad{1}+permute(decodeGrad{1}, [2, 1, 3]);
         end
         grad = [grad, encodeGrad(2:end), decodeGrad(2:end)];
      end
      
      function loss = compute_loss(obj, batch)
         x = batch{1};
         xRecon = obj.output(x);
         loss = obj.compute_loss_from_output(xRecon, x);
      end
      
      function loss = compute_loss_from_output(obj, xRecon, x)
         loss = obj.decodeLayer.compute_loss(xRecon, x);
      end
      
      function h = encode(obj, x)
         h = obj.encodeLayer.feed_forward(x);
      end
      
      function xRecon = output(obj, x)
         h = obj.encodeLayer.feed_forward(x);
         xRecon = obj.decodeLayer.feed_forward(h);
      end
      
      function increment_params(obj, delta)
         obj.encodeLayer.increment_params(delta(1:obj.encodeGradSize));
         
         if length(delta) > 1
            obj.decodeLayer.increment_params([0, ...
                                              delta(obj.encodeGradSize+1:end)]);
         end
         
         if obj.isTiedWeights
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         end
      end
      
      function gather(obj)
         obj.encodeLayer.gather();
         obj.decodeLayer.gather();
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.encodeLayer.push_to_GPU();
         obj.decodeLayer.push_to_GPU();
         obj.gpuState.isGPU = true;
      end
      
      function reset(obj)
         obj.encodeLayer.init_params();
         obj.decodeLayer.init_params();
         if obj.isTiedWeights 
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         end
      end
      
      function pTrans = get_encode_params_transposed(obj)
         if ndims(obj.encodeLayer.params{1}) <= 2
            pTrans = obj.encodeLayer.params{1}';
         else
            pTrans = permute(obj.encodeLayer.params{1}, [2, 1, 3]);
         end
      end
      
      function objCopy = copy(obj)
         objCopy = AutoEncoder();
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.gpuState = obj.gpuState;
      end
   end
   
end

