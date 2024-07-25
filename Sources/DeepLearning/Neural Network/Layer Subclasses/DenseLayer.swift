//
//  DenseLayer.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 16/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

internal class DenseLayer: Layer {
    
    var weights: Tensor<Float>
    var biases: Tensor<Float>
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        assert(inputs.count == self.inputAmount, "Incorrect amount of Inputs")

        self.zValues = []
        self.aValues = []
        self.newZValues = []
        
        for input in inputs {
            
            assert(input.shape == self.inputShape)
            
            self.zValues.append(input)
            self.aValues.append((self.weights * self.zValues.last! + self.biases).fast())
            
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
            zDerivative.append((self.weights.transpose() * aDerivative[i]).fast())
            
            /*aDerivative.append((self.weights.transpose() * derivatives[i]).fast())
            wDerivative.append((derivatives[i] * self.aValues[i]).fast())
            zDerivative.append(self.derivativeOfActivationFunction(aDerivative[i]))*/

        }
            
        let finalW = wDerivative.reduce(Tensor<Float>(self.weights.shape, constant: 0)) { ($0 + $1).fast() }
        let finalB = aDerivative.reduce(Tensor<Float>(self.biases.shape, constant: 0)) { ($0 + $1).fast() }

        return (newDerivatives: zDerivative, internalDerivatives: [finalW, finalB])
        
    }
    
    init(inputs: Int, outputs: Int, activationFunction: ActivationFunction, standardDeviation sdv: Float) {
        
        let inputShape = Shape([Dimension(inputs)])
        let outputShape = Shape([Dimension(outputs)])
        
        self.weights = Tensor<Float>(Shape([Dimension(outputs), Dimension(inputs)]), standardDeviation: sdv)
        self.biases = Tensor<Float>(Shape([Dimension(outputs)]), constant: 0)
        
        super.init(inputShape: inputShape, outputShape: outputShape, activationFunction: activationFunction)
                
        self.derivativeAmount = 2
        
    }
    
    override func applyDerivatives(_ derivatives: [Tensor<Float>], learningRate: Float) {
        
        assert(derivatives.count == self.derivativeAmount, "Incorrect amount of Derivatives")
        
        self.weights = (self.weights + (learningRate * derivatives[0])).fast()
        self.biases = (self.biases + (learningRate * derivatives[1])).fast()

    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let weights = try values.decode(Tensor<Float>.self, forKey: .weights)
        let biases = try values.decode(Tensor<Float>.self, forKey: .biases)
        let activationFunction = try values.decode(ActivationFunction.self, forKey: .activationFunction)
        
        self.weights = weights
        self.biases = biases
        
        super.init(inputShape: Shape([weights.shape[1]]), outputShape: biases.shape, activationFunction: activationFunction)
        
        assert(biases.shape.dimensions.count == 1)
        
        self.derivativeAmount = 2

    }
    
    override func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.weights, forKey: .weights)
        try container.encode(self.activationFunction, forKey: .activationFunction)
        try container.encode(self.biases, forKey: .biases)

        try container.encode(Network.InternalLayerType.dense, forKey: .type)

    }
    
    override func copy() -> DenseLayer {
        
        let newLayer = DenseLayer(inputs: self.inputShape.volume, outputs: self.outputShape.volume, activationFunction: self.activationFunction, standardDeviation: 0)
        newLayer.weights = self.weights
        newLayer.biases = self.biases
        
        return newLayer
        
    }
    
    private enum CodingKeys: String, CodingKey {
        
        case weights
        case biases
        case activationFunction
        case layerType
        case type

    }
    
}
