classdef FFNInspector < Model
   
   properties
      ffn
      xMax
      inputSize
   end
   
   methods
      function obj = FFNInspector(ffn, layerIdx, inputSize)
         if nargin > 0
            obj.ffn = ffn.copy();
            obj.ffn.hiddenLayers = obj.ffn.hiddenLayers(1:layerIdx);
            obj.ffn.outputLayer = [];
            obj.ffn.inputDropout = 0;
            obj.ffn.hiddenDropout = 0;
            obj.inputSize = inputSize;
         end
      end
      
      
      function focus(obj, hiddenIdx)
         % Change focus to inspect hiddenIdx unit in penultimate layer. Also
         % randomly initializes inspector parameters (ffn inputs).
 
         obj.initialize_xMax();
         obj.ffn.outputLayer = InspectorOutputLayer(hiddenIdx, obj.ffn.gpuState);
      end
      
      
      function initialize_xMax(obj)
         obj.xMax = obj.ffn.gpuState.zeros(obj.inputSize);
      end
      
      
      function [grad, output] = gradient(obj, ~)
         [~, output, dLdx] = obj.ffn.gradient({obj.xMax, []});
         grad = {dLdx};
      end
      
      
      function y = output(obj, x)
         y = obj.ffn.output(x);
      end
      
      
      function loss = compute_loss(obj, ~)
         loss = obj.ffn.compute_loss({obj.xMax, []});
      end
      
      
      function loss = compute_loss_from_output(obj, y, ~)
         loss = obj.ffn.compute_loss_from_output(y, []);
      end
      
      
      function increment_params(obj, delta)
         obj.xMax = obj.xMax + delta{1};
      end
      
      
      function gather(obj)
         obj.ffn.gather();
         obj.xMax = gather(obj.xMax);
      end
      
      
      function push_to_GPU(obj)
         obj.ffn.push_to_GPU();
         obj.xMax = single(gpuArray(obj.xMax));
      end
      
      
      function objCopy = copy(obj)
         objCopy = FFNInspector();
         objCopy.ffn = obj.ffn.copy();
         objCopy.xMax = obj.xMax;         
         objCopy.inputSize = obj.inputSize;
      end
      
      
      function reset(~)
         % pass
      end

   end   
end

