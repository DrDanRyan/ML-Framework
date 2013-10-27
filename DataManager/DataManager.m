classdef DataManager < matlab.mixin.Copyable
   
   properties
      trainingData % cell array of data relevant for training; targets should be last entry if present
      validationData % cell array of data relevant for validation; same shape as trainingData
      
      batchSize
      trainingSize
      startIdx
      stopIdx
   end
   
   methods
      function obj = DataManager(trainingData, validationData, varargin)
         
         if nargin == 0
            return
         end
         
         obj.trainingData = trainingData; 
         obj.validationData = validationData;
         
         p = inputParser();
         p.addParamValue('batchSize', []);
         parse(p, varargin{:});
         
         if ~isempty(p.Results.batchSize) % Use mini-batches; set batchsize and trainingSize
            p.batchSize = p.Results.batchSize;
            obj.trainingSize = size(trainingData{1}, 2);
            obj.shuffle_training_data();
         end
      end
      
      function [batch, endOfEpochFlag] = next_batch(obj)
         if isempty(obj.batchSize) % full batch
            batch = obj.trainingData;
            endOfEpochFlag = true;
         else % mini-batch
            batch = cellfun(@(v) v(:,obj.startIdx:obj.stopIdx), 'UniformOutput', false);
            
            if obj.stopIdx == obj.trainingSize
               endOfEpochFlag = true;
               obj.shuffle_training_data();
            else
               endOfEpochFlag = false;
               obj.startIdx = obj.startIdx + obj.batchSize;
               obj.stopIdx = min(obj.trainingSize, obj.stopIdx + obj.batchSize);
            end
         end
      end
      
      function shuffle_training_data(obj)
         permvec = randperm(obj.trainingSize);
         obj.trainingData = cellfun(@(v) v(:,permvec), obj.trainingData, ...
                                                      'UniformOutput', false);
         obj.startIdx = 1;
         obj.stopIdx = min(obj.batchSize, obj.trainingSize);
      end
      
      function reset(obj)
         if ~isempty(obj.batchSize)
            obj.trainingSize = size(obj.trainingData{1}, 2);
            obj.shuffle_training_data();
         end
      end
   end
end