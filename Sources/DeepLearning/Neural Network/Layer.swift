//
//  Layer.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

internal class Layer: Codable, Differentiable {
   
    internal var derivativeAmount: Int = 0
        
    var inputShape: Shape
    var inputAmount: Int = 1
    
    var outputShape: Shape
    var outputAmount: Int = 1

    var zValues: [Tensor<Float>]
    var aValues: [Tensor<Float>]
    
    var newZValues: [Tensor<Float>]
    
    var activationFunction:ActivationFunction
    
    func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        precondition(inputs.count == self.inputAmount, "Incorrect amount of Inputs")
        
        self.zValues = []
        self.aValues = []
        self.newZValues = []
                
        for input in inputs {
        
            assert(input.shape == self.inputShape, "Incorrect Input Shape")
            
            self.zValues.append(input)
            self.aValues.append(input)
            self.newZValues.append(self.applyActivationFunction(self.aValues.last!))
        
        }
            
        return self.newZValues
        
    }
    
    func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        precondition(derivatives.count == self.outputAmount, "Incorrect amount of Derivatives")
        
        var newDerivative:[Tensor<Float>] = []
        
        for derivative in derivatives {
            
            newDerivative.append(self.derivativeOfActivationFunction(derivative))
            
        }
        
        return (newDerivatives: newDerivative, internalDerivatives: [])
        
    }
    
    func applyActivationFunction(_ zValues: Tensor<Float>) -> Tensor<Float> {
        
        var tensor = zValues
        
        switch self.activationFunction {
            
        case .normal: return tensor
        case .positive: tensor.elements = tensor.elements.map { if $0 >= 0 { return 1 } else { return 0 } }
        case .relu: tensor.elements = tensor.elements.map { return max(0, $0) }
        case .smoothRelu: tensor.elements = tensor.elements.map { return log(1+exp($0)) }
        case .leakyRelu: tensor.elements = tensor.elements.map { return max(0.1 * $0, $0) }
        case .sigmoid: tensor.elements = tensor.elements.map { return 1/(1+exp(-$0)) }
        case .arctan: tensor.elements = tensor.elements.map { return atan($0)/(.pi/2) }
        case .tanh: tensor.elements = tensor.elements.map { return tanh($0) }
            
        }
        
        return tensor
        
    }
    
    func derivativeOfActivationFunction(_ derivative: Tensor<Float>) -> Tensor<Float> {
        
        precondition(derivative.shape == self.outputShape, "Derivatives are of the wrong Shape")
        
        var tensor = Tensor<Float>(derivative.shape, constant: 0)
        
        switch self.activationFunction {
            
        case .normal: return derivative
        case .positive: fatalError("not implemented")//return tensor
        case .relu: tensor.elements = self.aValues[0].elements.map { if $0 >= 0 { return 1 } else { return 0 } }
        case .smoothRelu: tensor.elements = self.aValues[0].elements.map { return 1/(1+exp(-$0)) }
        case .leakyRelu: tensor.elements = self.aValues[0].elements.map { if $0 >= 0 { return 1 } else { return 0.1 } }
        case .sigmoid: tensor.elements = self.aValues[0].elements.map { let sigmoid = 1/(1+exp(-$0)); return sigmoid * (1 - sigmoid) }
        case .arctan: tensor.elements = self.aValues[0].elements.map { return 2/(.pi * (1 + ($0 * $0))) }
        case .tanh: tensor.elements = self.aValues[0].elements.map { return 1 - (tanh($0) * tanh($0)) }
            
        }
        
        return (tensor ** derivative).fast()
        
    }
    
    init(inputShape: Shape, outputShape: Shape, activationFunction: ActivationFunction) {
        
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.activationFunction = activationFunction
        self.zValues = []
        self.aValues = []
        self.newZValues = []

    }
    
    func addDerivatives(_ firstDerivative: [Tensor<Float>], _ secondDerivative: [Tensor<Float>]) -> [Tensor<Float>] {
        
        precondition(firstDerivative.count == secondDerivative.count, "Derivatives need to be of the same Shape")
        
        var output:[Tensor<Float>] = []
        
        for i in firstDerivative.indices {
            
            precondition(firstDerivative[i].shape == secondDerivative[i].shape, "Derivatives need to be of the same Shape")
            
            output.append((firstDerivative[i] + secondDerivative[i]).fast())
            
        }
        
        return output
        
    }
    
    func scaleDerivatives(_ derivative: [Tensor<Float>], factor: Float) -> [Tensor<Float>] {
        
        let output:[Tensor<Float>] = derivative.map { return (factor * $0).fast() }
        
        return output
        
    }
    
    internal func averageDerivatives(_ derivatives: [[Tensor<Float>]]) -> [Tensor<Float>] {
        
        if derivatives.count == 0 { return [] }
        if derivatives.count == 1 { return derivatives.first! }
        
        let factor = 1/Float(derivatives.count)
        
        var output:[Tensor<Float>] = derivatives.first!
        for (i, t) in output.enumerated() {
            output[i] = (factor * t).fast()
        }
        
        if derivatives.count == 1 { return output }
        
        for i in 1..<derivatives.count {
                        
            precondition(derivatives[i].count == output.count, "Derivatives need to be of the same Shape")
            
            for j in derivatives[i].indices {
                
                precondition(derivatives[i][j].shape == output[j].shape, "Derivatives need to be of the same Shape")
                
                output[j] = (factor * derivatives[i][j] + output[j]).fast()
                
            }
            
        }
        
        return output
        
    }
    
    func applyDerivatives(_ derivatives: [Tensor<Float>], learningRate: Float) {
        
        precondition(derivatives.count == self.derivativeAmount, "Incorrect amount of Derivatives")
        
        return
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let shape = try values.decode(Shape.self, forKey: .shape)
        let activationFunction = try values.decode(ActivationFunction.self, forKey: .activationFunction)

        self.inputShape = shape
        self.outputShape = shape
        self.activationFunction = activationFunction
        self.zValues = []
        self.aValues = []
        self.newZValues = []
        
    }
    
    func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.inputShape, forKey: .shape)
        try container.encode(self.activationFunction, forKey: .activationFunction)

        try container.encode(Network.InternalLayerType.activation, forKey: .type)

    }
    
    internal func copy() -> Layer {
        
        let newLayer = Layer(inputShape: self.inputShape, outputShape: self.outputShape, activationFunction: self.activationFunction)
        newLayer.inputAmount = self.inputAmount
        newLayer.outputAmount = self.outputAmount
        
        return newLayer
        
    }
    
    private enum CodingKeys: String, CodingKey {
        
        case shape
        case activationFunction
        case type
        
    }
    
}

public enum LayerType {
    
    case activation(activationF: ActivationFunction)
    case dense(outputs: Int, activation: ActivationFunction)
    case softmax
    case reshape(newShape: Shape)
    case flatten
    case convolutional(filterSize: [Int], filterStride: [Int], numberOfFilters: Int, activation: ActivationFunction)
    case maxPooling(poolingSize: [Int], poolingStride: [Int])
    case encoder(outputs: Int)
    case decoder(encoderIndex: Int)
    
}
