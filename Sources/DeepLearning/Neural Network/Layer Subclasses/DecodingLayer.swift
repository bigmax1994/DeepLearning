//
//  File.swift
//  
//
//  Created by Max Gasslitter Strobl on 15/09/22.
//

import Foundation

class DecodingLayer: DenseLayer {
    
    let uuid: UUID
    var encoder: EncodingLayer?
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        assert(inputs.count == self.inputAmount, "Incorrect amount of Inputs")

        self.zValues = []
        self.aValues = []
        self.newZValues = []
        
        for input in inputs {
            
            assert(input.shape == self.inputShape)
            
            self.zValues.append(input)
            self.aValues.append((self.encoder!.weights.transpose() * self.zValues.last!).fast())
            
            self.newZValues.append(self.applyActivationFunction(self.aValues.last!))
        
        }
        
        return self.newZValues
        
    }
    
    override func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        assert(derivatives.count == self.outputAmount, "Incorrect amount of Derivatives")
                
        var aDerivative:[Tensor<Float>] = []
        var wDerivative:[Tensor<Float>] = []
        var zDerivative:[Tensor<Float>] = []
        
        for i in derivatives.indices {

            aDerivative.append(self.derivativeOfActivationFunction(derivatives[i]))
            wDerivative.append((aDerivative[i] * self.zValues[i]).fast())
            zDerivative.append((self.encoder!.weights * aDerivative[i]).fast())
            
            /*aDerivative.append((self.encoder!.weights * derivatives[i]).fast())
            wDerivative.append((derivatives[i] * self.aValues[i]).fast())
            zDerivative.append(self.derivativeOfActivationFunction(aDerivative[i]*/

        }
            
        let finalW = wDerivative.reduce(Tensor<Float>(self.encoder!.weights.shape.transposed(), constant: 0)) { ($0 + $1).fast() }

        return (newDerivatives: zDerivative, internalDerivatives: [finalW])
        
    }
    
    override func applyDerivatives(_ derivatives: [Tensor<Float>], learningRate: Float) {
        
        assert(derivatives.count == self.derivativeAmount, "Incorrect amount of Derivatives")
        
        self.encoder!.weights = (self.encoder!.weights + (learningRate * derivatives[0].transpose())).fast()

    }
    
    init(encoder: EncodingLayer) {
        
        self.encoder = encoder
        self.uuid = encoder.uuid
        
        super.init(inputs: encoder.outputShape.volume, outputs: encoder.inputShape.volume, activationFunction: .sigmoid, standardDeviation: 0)
        
        self.derivativeAmount = 1
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.uuid = try values.decode(UUID.self, forKey: .uuid)
        self.encoder = nil
        
        try super.init(from: decoder)
        
        self.derivativeAmount = 1
        
    }
    
    override func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.encoder!.uuid, forKey: .uuid)

        try super.encode(to: encoder)
        
        try container.encode(Network.InternalLayerType.decoder, forKey: .type)
        
    }
    
    override func copy() -> DecodingLayer {
        
        let newLayer = DecodingLayer(encoder: self.encoder!)
        
        return newLayer
        
    }
    
    private enum CodingKeys: String, CodingKey {
        
        case uuid
        case type
        
    }
    
}
