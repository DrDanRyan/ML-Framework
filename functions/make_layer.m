function layer = make_layer(typeString, args)

switch typeString
   case 'LogisticHidden'
      layer = LogisticHiddenLayer(args{:});
   case 'TanhHidden'
      layer = TanhHiddenLayer(args{:});
   case 'ReluHidden'
      layer = ReluHiddenLayer(args{:});
   case 'MaxoutHidden'
      layer = MaxoutHiddenLayer(args{:});
   case 'LogisticOutput'
      layer = LogisticOutputLayer(args{:});
   case 'SVMOutput'
      layer = SVMOutputLayer(args{:});
end


end

