classdef ImputingDataManager < DataManager
   % Keeps track of NaN values in trainingData as well as most recently
   % estimated values imputed by model during training.   
   
   properties
      % Logical array same size as trainingData indicating
      % where trainingData had NaN values originally
      isNaN       
      stopIdx  % stopIdx used to generate previous mini-batch
   end
   
   methods
      function obj = ImputingDataManager(trainingData, validationData, varargin)
         obj = obj@DataManager(trainingData, validationData, varargin{:});
         obj.isNaN = isnan(obj.trainingData{1});
         obj.trainingData{1}(obj.isNaN) = 0;
      end
      
      function batch = next_batch(obj)
         % Passes logical isNaN array along with training data batch.
         
         if isempty(obj.batchSize) % full batch
            batch = obj.trainingData;
            batch{end+1} = obj.isNaN;
         else % mini-batch
            obj.stopIdx = min(obj.trainingSize, ...
                              obj.batchIdx + obj.batchSize - 1);
            batch = cellfun(@(v) v(:,obj.batchIdx:obj.stopIdx,:,:), ...
                              obj.trainingData, 'UniformOutput', false);
            batch{end+1} = obj.isNaN(:,obj.batchIdx:obj.stopIdx,:,:);
         end  
      end
      
      function update_imputed_data(obj, xNew)
         % Overwrites NaN (or previously imputed values) with newly imputed
         % values from last model update.
         
         if isempty(obj.batchSize) % full batch
            obj.trainingData{1}(obj.isNaN) = xNew;
         else % mini-batch
            obj.trainingData{1}(...
               obj.isNaN(:,obj.batchIdx:obj.stopIdx,:,:)) = xNew;
            
            if obj.stopIdx == obj.trainingSize
               obj.shuffle_training_data();
            else
               obj.batchIdx = obj.stopIdx + 1;
            end
         end
      end
      
      function shuffle_training_data(obj)
         % Shuffles isNaN matrix along with trainingData.
         
         permvec = randperm(obj.trainingSize);
         obj.trainingData = cellfun(@(v) v(:,permvec,:,:), obj.trainingData, ...
                                                      'UniformOutput', false);
                                  
         if ~isempty(obj.isNaN) 
            obj.isNaN = obj.isNaN(:,permvec,:,:);
         end
         obj.batchIdx = 1;
      end
      
      function reset(obj)
         % Sets all NaN values to zero during reset.
         
         reset@DataManager(obj);
         obj.trainingData{1}(obj.isNaN) = 0;
      end
      
   end
end