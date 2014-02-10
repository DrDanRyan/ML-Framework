classdef DataManager < matlab.mixin.Copyable
   
   properties
      trainingData % cell array of data relevant for training; targets should be last entry if present
      validationData % cell array of data relevant for validation; same shape as trainingData
      batchSize   % Size of batch used for training
      lossBatchSize % Size of batch used to compute training and validation losses
      trainLossSampleSize % number of training examples to sample for training loss
      trainLossSample % a sample of the training data used to estimate loss on training set
      trainingSize % number of training examples total
      validationSize % number of validation examples total
      
      % Used to keep track of batches
      batchIdx
      lossIdx
   end
   
   methods
      function obj = DataManager(trainingData, validationData, varargin)
         
         if nargin == 0
            return
         end
         
         obj.trainingData = trainingData; 
         obj.trainingSize = size(trainingData{1}, 2);
         obj.validationData = validationData;
         if ~isempty(validationData)
            obj.validationSize = size(validationData{1}, 2);
         else
            obj.validationSize = 0;
         end
         
         p = inputParser();
         p.addParamValue('batchSize', []);
         p.addParamValue('lossBatchSize', []);
         p.addParamValue('trainLossSampleSize', []);
         parse(p, varargin{:});
         
         obj.batchSize = p.Results.batchSize;
         obj.lossBatchSize = p.Results.lossBatchSize;
         obj.trainLossSampleSize = p.Results.trainLossSampleSize;
         
         if ~isempty(p.Results.batchSize) % Using mini-batches => shuffle data before beginning
            obj.shuffle_training_data();
         end
         
         if ~isempty(obj.lossBatchSize)
            obj.lossIdx = 1;
         end
      end
      
      function batch = next_batch(obj)
         if isempty(obj.batchSize) % full batch
            batch = obj.trainingData;
         else % mini-batch
            stopIdx = min(obj.trainingSize, obj.batchIdx + obj.batchSize - 1);
            batch = cellfun(@(v) v(:,obj.batchIdx:stopIdx,:,:), obj.trainingData, ...
                              'UniformOutput', false);
            
            if stopIdx == obj.trainingSize
               obj.shuffle_training_data();
            else
               obj.batchIdx = stopIdx + 1;
            end
         end
      end
      
      function [batch, isContinue] = trainLoss_batch(obj)
         if ~isempty(obj.trainLossSampleSize) && isempty(obj.trainLossSample)
            permvec = randperm(obj.trainingSize, obj.trainLossSampleSize);
            obj.trainLossSample = cellfun(@(v) v(:,permvec,:,:), obj.trainingData, ...
                                             'UniformOutput', false);
         end
         
         if isempty(obj.lossBatchSize) % no batches, just give whole set
            if isempty(obj.trainLossSample)
               batch = obj.trainingData;
            else
               batch = obj.trainLossSample;
               obj.trainLossSample = [];
            end
            isContinue = false;
         else % using batches
            if isempty(obj.trainLossSample)
               stopIdx = min(obj.trainingSize, obj.lossIdx + obj.lossBatchSize - 1);
               batch = cellfun(@(v) v(:,obj.lossIdx:stopIdx,:,:), obj.trainingData, ...
                                 'UniformOutput', false);
                              
               if stopIdx == obj.trainingSize
                  isContinue = false;
                  obj.lossIdx = 1;
               else
                  isContinue = true;
                  obj.lossIdx = stopIdx + 1;
               end               
            else
               stopIdx = min(obj.trainLossSampleSize, obj.lossIdx + obj.lossBatchSize - 1);
               batch = cellfun(@(v) v(:,obj.lossIdx:stopIdx,:,:), obj.trainLossSample, ...
                                 'UniformOutput', false);
                              
               if stopIdx == obj.trainLossSampleSize
                  isContinue = false;
                  obj.lossIdx = 1;
                  obj.trainLossSample = [];
               else
                  isContinue = true;
                  obj.lossIdx = stopIdx + 1;
               end               
            end
         end
      end
      
      function [batch, isContinue] = validLoss_batch(obj)
         if isempty(obj.lossBatchSize)
            batch = obj.validationData;
            isContinue = false;
         else
            stopIdx = min(obj.validationSize, obj.lossIdx + obj.lossBatchSize - 1);
            batch = cellfun(@(v) v(:,obj.lossIdx:stopIdx,:,:), obj.validationData, ...
                                 'UniformOutput', false);

            if stopIdx == obj.validationSize
               isContinue = false;
               obj.lossIdx = 1;
            else
               isContinue = true;
               obj.lossIdx = stopIdx + 1;
            end
         end
      end
      
      function shuffle_training_data(obj)
         permvec = randperm(obj.trainingSize);
         obj.trainingData = cellfun(@(v) v(:,permvec,:,:), obj.trainingData, ...
                                                      'UniformOutput', false);
         obj.batchIdx = 1;
      end
      
      function reset(obj)
         obj.trainingSize = size(obj.trainingData{1}, 2);
         if ~isempty(obj.validationData)
            obj.validationSize = size(obj.validationData{1}, 2);
         else
            obj.validationSize = 0;
         end
         
         if ~isempty(obj.lossBatchSize)
            obj.lossIdx = 1;
         end
         
         if ~isempty(obj.batchSize)
            obj.shuffle_training_data();
         end
      end
   end
end