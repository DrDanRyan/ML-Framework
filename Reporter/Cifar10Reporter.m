classdef Cifar10Reporter < Reporter
   % A reporter class for visualizing Cifar10 filters (not convolutional
   % filters) during training
   
   properties
      consoleReporter
      rows
      cols
   end
   
   methods
      function obj = Cifar10Reporter(rows, cols)
         obj.consoleReporter = ConsoleReporter();
         obj.rows = rows;
         obj.cols = cols;
         obj.init_figure();
      end
      
      function init_figure(obj)
         for i = 1:obj.rows
            for j = 1:obj.cols
               subtightplot(obj.rows, obj.cols, obj.cols*(i-1) + j, ...
                              [.01, .01]);
               image(zeros(32, 32), 'CDataMapping', 'scaled');
               colormap('gray');
               axis off tight
            end
         end
         drawnow;
      end
      
      function report(obj, progressMonitor, model)
         if isa(model, 'AutoEncoder')
            layer = model.encodeLayer;
         else % Assume FeedForwardNet
            layer = model.hiddenLayers{1};
         end
         
         if isa(layer, 'CompositeHiddenLayer')
            W = layer.layers{1}.params{1};
         else % assume basic HiddenLayer
            W = layer.params{1};
         end
         
         mins = min(W, [], 2);
         maxs = max(W, [], 2);
         W = bsxfun(@rdivide, W, maxs - mins);
         W = bsxfun(@minus, W, min(W, [], 2));
         W = reshape(W, [], 32, 32, 3);
         
         for i = 1:obj.rows
            for j = 1:obj.cols
               idx = obj.cols*(i-1) + j;
               subtightplot(obj.rows, obj.cols, idx, [.01, .01]);
               f = squeeze(W(idx,:,:,:));
               f = min(1, max(f, 0));
               image(f);
               axis off tight
            end
         end
         drawnow;
         obj.consoleReporter.report(progressMonitor, model);
      end
      
      function reset(obj)
         close;
         obj.init_figure();
      end
      
   end
end

