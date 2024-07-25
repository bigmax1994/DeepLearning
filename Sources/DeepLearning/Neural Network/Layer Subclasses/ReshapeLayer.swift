//
//  ReshapeLayer.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 16/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

final internal class ReshapeLayer: Layer {
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        assert(inputs.count == self.inputAmount, "Incorrect amount of Inputs")
        
        self.zValues = inputs
        self.aValues = self.zValues.map { return Tensor<Float>.reshaped($0, to: self.outputShape) }
        self.newZValues = self.aValues
        
        return self.aValues
        
    }
    
    override func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        assert(derivatives.count == self.outputAmount, "Derivatives are of the Incorrect Shape")
        
        return (newDerivatives: derivatives.map { return Tensor<Float>.reshaped($0, to: self.inputShape) }, internalDerivatives: [])
        
    }
    
    init(inputShape: Shape, outputShape: Shape) {
        
        assert(inputShape.volume == outputShape.volume, "Shapes do not have the same volume")
        
        super.init(inputShape: inputShape, outputShape: outputShape, activationFunction: .normal)
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let inputShape = try values.decode(Shape.self, forKey: .inputShape)
        let outputShape = try values.decode(Shape.self, forKey: .outputShape)

        assert(inputShape.volume == outputShape.volume, "Shapes do not have the same volume")
        
        super.init(inputShape: inputShape, outputShape: outputShape, activationFunction: .normal)

        
    }
    
    override func encode(to encoder: Encoder) throws {
     
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.inputShape, forKey: .inputShape)
        try container.encode(self.outputShape, forKey: .outputShape)
        
        try container.encode(Network.InternalLayerType.reshape, forKey: .type)

    }
    
    override func copy() -> ReshapeLayer {
        
        let newLayer = ReshapeLayer(inputShape: self.inputShape, outputShape: self.outputShape)
        
        return newLayer
        
    }
    
    private enum CodingKeys: String, CodingKey {
        
        case inputShape
        case outputShape
        case type

    }
    
}
