//
//  Encoder.swift
//  
//
//  Created by Max Gasslitter Strobl on 15/09/22.
//

import Foundation

internal class EncodingLayer: DenseLayer {
    
    let uuid: UUID
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        assert(inputs.count == self.inputAmount, "Incorrect amount of Inputs")

        self.zValues = []
        self.aValues = []
        self.newZValues = []
        
        for input in inputs {
            
            assert(input.shape == self.inputShape)
            
            self.zValues.append(input)
            self.aValues.append(input)
            
            self.newZValues.append((self.weights * self.aValues.last!).fast())
        
        }
        
        return self.newZValues
        
    }
    
    override func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        assert(derivatives.count == self.outputAmount, "Incorrect amount of Derivatives")
                
        var aDerivative:[Tensor<Float>] = []
        var wDerivative:[Tensor<Float>] = []
        var zDerivative:[Tensor<Float>] = []
        
        for i in derivatives.indices {
            
            aDerivative.append(derivatives[i])
            wDerivative.append((aDerivative[i] * self.zValues[i]).fast())
            zDerivative.append((self.weights.transpose() * aDerivative[i]).fast())

            /*aDerivative.append((self.weights.transpose() * derivatives[i]).fast())
            wDerivative.append((derivatives[i] * self.aValues[i]).fast())*/

        }
            
        let finalW = wDerivative.reduce(Tensor<Float>(self.weights.shape, constant: 0)) { ($0 + $1).fast() }

        return (newDerivatives: zDerivative, internalDerivatives: [finalW])
        
    }
    
    override func applyDerivatives(_ derivatives: [Tensor<Float>], learningRate: Float) {
        
        assert(derivatives.count == self.derivativeAmount, "Incorrect amount of Derivatives")
        
        self.weights = (self.weights + (learningRate * derivatives[0])).fast()

    }
    
    init(inputs: Int, outputs: Int, standardDeviation sdv: Float) {
        
        self.uuid = UUID()
        
        super.init(inputs: inputs, outputs: outputs, activationFunction: .normal, standardDeviation: sdv)
        
        self.derivativeAmount = 1
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.uuid = try values.decode(UUID.self, forKey: .uuid)
        
        try super.init(from: decoder)
        
        self.derivativeAmount = 1
        
    }
    
    override func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.uuid, forKey: .uuid)
        
        try super.encode(to: encoder)
        
        try container.encode(Network.InternalLayerType.encoder, forKey: .type)

    }
    
    override func copy() -> EncodingLayer {
        
        let newLayer = EncodingLayer(inputs: self.inputShape.volume, outputs: self.outputShape.volume, standardDeviation: 0)
        newLayer.weights = self.weights
        newLayer.biases = self.biases
        
        return newLayer
        
    }
    
    private enum CodingKeys: String, CodingKey {
        
        case uuid
        case type
        
    }
    
}
