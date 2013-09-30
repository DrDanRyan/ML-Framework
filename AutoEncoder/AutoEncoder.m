classdef AutoEncoder < handle
   % Generic AutoEncoder
   
   properties
      encodeLayer % a HiddenLayer object that functions as the encoding layer
      decodeLayer % a OutputLayer object that functions as the decoding layer and loss function
      isTiedWeights % a boolean indicating if the params in encodeLayer and decodeLayer are shared
   end
   
   methods
      function grad = gradient(obj, x)
         
      end
      
      function update_params(obj, delta_params)

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
         
      end
   end
   
end

