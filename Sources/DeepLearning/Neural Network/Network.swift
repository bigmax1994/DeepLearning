//
//  Network.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

public class Network: Codable, Differentiable {
    
    var derivativeAmount: Int { get { return layers.reduce(0) { $0 + $1.derivativeAmount } } }
    
    public var inputShape: Shape
    public var inputAmount: Int
    
    public var outputShape: Shape
    public var outputAmount: Int
    
    public var aValues: [Tensor<Float>]
    public var zValues: [Tensor<Float>]
    
    private var layers: [Layer]
    
    public var lastLoss: Float = 0
    public var lossFunction: LossFunction
    public func print() {
        Swift.print("weight: \((self.layers[0] as! DenseLayer).weights.elements[0])")
    }
    
    public func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        self.zValues = inputs
                                        
        var lastOutput:[Tensor<Float>] = inputs
            
        for layer in self.layers {
                
            lastOutput = layer.propagateForeward(inputs: lastOutput)
                
        }
        
        self.aValues = lastOutput
        
        return self.aValues
                    
    }
    
    internal func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        var lastDerivative = derivatives
        var internalDerivatives:[Tensor<Float>] = []
        
        for layer in self.layers.reversed() {
            
            let derivative = layer.propagateBackward(derivatives: lastDerivative)
            
            internalDerivatives.append(contentsOf: derivative.internalDerivatives.reversed())
            lastDerivative = derivative.newDerivatives
            
        }
        
        return (newDerivatives: lastDerivative, internalDerivatives: internalDerivatives.reversed())
        
    }
    
    public func calculateWithCost(inputs: [Tensor<Float>], desiredOutputs: [Tensor<Float>]) -> (outputs: [Tensor<Float>], loss: Float) {
        
        let outputs = self.propagateForeward(inputs: inputs)
        let loss = self.applyLossFunction(outputs, desiredOutputs: desiredOutputs)
        
        return (outputs: outputs, loss: loss)
        
    }
    
    public func getDerivatives(inputs: [Tensor<Float>], desiredOutputs: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>], outputs: [Tensor<Float>], loss: Float) {
        
        //var start = CFAbsoluteTimeGetCurrent()
        let outputs = self.calculateWithCost(inputs: inputs, desiredOutputs: desiredOutputs)
        //var diff = CFAbsoluteTimeGetCurrent() - start
        //Swift.print("calculation took \(diff)")
        
        //start = CFAbsoluteTimeGetCurrent()
        let finalDerivative = derivativeOfLossFunction(outputs.outputs, desiredOutputs: desiredOutputs)
        let derivatives = propagateBackward(derivatives: finalDerivative)
        //diff = CFAbsoluteTimeGetCurrent() - start
        //Swift.print("derivation took \(diff)")
        
        return (newDerivatives: derivatives.newDerivatives, internalDerivatives: derivatives.internalDerivatives, outputs: outputs.outputs, loss: outputs.loss)
        
    }
    
    private func applyLossFunction(_ outputs: [Tensor<Float>], desiredOutputs: [Tensor<Float>]) -> Float {
        
        var loss:Float = 0
        
        precondition(outputs.count == desiredOutputs.count, "Desired Outputs are not of the correct Shape")
        
        switch self.lossFunction {
            
        case .meanErrorSquared: for i in 0..<outputs.count {
                
                precondition(outputs[i].volume == desiredOutputs[i].volume, "Desired Outputs are not of the correct Shape")
                
                for j in 0..<outputs[i].volume {
                    
                    loss += pow(outputs[i].elements[j] - desiredOutputs[i].elements[j], 2)
                    
                }
                
            }
            
        case .crossEntropy: for i in 0..<outputs.count {
            
                precondition(outputs[i].volume == desiredOutputs[i].volume, "Desired Outputs are not of the correct Shape")
                
                for j in 0..<outputs[i].volume {
                    
                    if outputs[i].elements[j] != 0 {
                        
                        loss += -desiredOutputs[i].elements[j] * log(outputs[i].elements[j])
                        
                    }else{
                        
                        loss += desiredOutputs[i].elements[j] * 1000000
                        
                    }
                    
                }
                
            }
            
        }
        
        return loss
        
    }
    
    private func derivativeOfLossFunction(_ outputs: [Tensor<Float>], desiredOutputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        var derivativeOutputs:[Tensor<Float>] = outputs
        
        precondition(outputs.count == desiredOutputs.count, "Desired Outputs are not of the correct Shape")
        
        switch self.lossFunction {
            
        case .meanErrorSquared: for i in 0..<outputs.count {
                
                precondition(outputs[i].volume == desiredOutputs[i].volume, "Desired Outputs are not of the correct Shape")
                
                for j in 0..<outputs[i].volume {
                    
                    derivativeOutputs[i].elements[j] = 2 * (outputs[i].elements[j] - desiredOutputs[i].elements[j])
                    
                }
                
            }
            
        case .crossEntropy: for i in 0..<outputs.count {
            
                precondition(outputs[i].volume == desiredOutputs[i].volume, "Desired Outputs are not of the correct Shape")
                
                for j in 0..<outputs[i].volume {
                    
                    if outputs[i].elements[j] != 0 {
                        
                        derivativeOutputs[i].elements[j] = -desiredOutputs[i].elements[j] * (1 / outputs[i].elements[j])
                        
                    }else{
                        
                        derivativeOutputs[i].elements[j] = 1000000
                        
                    }
                    
                }
                
            }
            
        }
        
        return derivativeOutputs
        
    }
    
    public func averageAndApplyDerivatives(_ derivatives: [[Tensor<Float>]], learningRate: Float) {
        
        let learningRate = abs(learningRate)
        
        let avrgDerivatives = self.averageDerivatives(derivatives)
        
        var counter = 0
        
        for layer in self.layers {
            
            let layerDerivatives = Array(avrgDerivatives[counter..<(counter + layer.derivativeAmount)])
            
            layer.applyDerivatives(layerDerivatives, learningRate: -learningRate)
            
            counter += layer.derivativeAmount
            
        }
        
    }
    
    internal func averageDerivatives(_ derivatives: [[Tensor<Float>]]) -> [Tensor<Float>] {
        
        let originalShapes = derivatives[0].map { return $0.shape }
        let mappedDerivative = derivatives.map { return $0.map { return Tensor<Float>.reshaped($0, to: Shape([$0.volume,1]))} }
        
        let factor = 1/Float(derivatives.count)
        
        var output:[Tensor<Float>] = mappedDerivative.first!
        
        if derivatives.count == 1 { return output }
        
        for i in 0..<derivatives.count {
                        
            precondition(mappedDerivative[i].count == output.count, "Derivatives need to be of the same Shape")
                        
            for j in derivatives[i].indices {
                
                precondition(mappedDerivative[i][j].shape == output[j].shape, "Derivatives need to be of the same Shape")
                
                output[j] = (factor * mappedDerivative[i][j] + output[j]).fast()
                
            }
            
        }
        
        output = output.enumerated().map { (i, value) in return Tensor<Float>.reshaped(value, to: originalShapes[i])}
        
        return output
        
    }
    
    public func add(_ newLayer: LayerType) {
        
        let lastShape = self.layers.last?.outputShape ?? self.inputShape
        
        switch newLayer {
            
        case let .activation( a ):
            self.layers.append(Layer(inputShape: lastShape,
                                     outputShape: lastShape,
                                     activationFunction: a))
        case let .dense( o , a ):
            precondition(lastShape.dimensions.count == 1, "Shape needs to be flattened for a Dense Layer")
            let sdv = sqrt(2/Float(lastShape[0].value))
            self.layers.append(DenseLayer(inputs: lastShape[0].value,
                                          outputs: o,
                                          activationFunction: a,
                                          standardDeviation: sdv))
        case .softmax:
            precondition(lastShape.dimensions.count == 1, "Shape needs to be flattened for a Softmax Layer")
            self.layers.append(SoftmaxLayer(features: lastShape[0].value))
        case let .reshape( o ):
            self.layers.append(ReshapeLayer(inputShape: lastShape,
                                            outputShape: o))
        case .flatten:
            self.layers.append(ReshapeLayer(inputShape: lastShape,
                                            outputShape: Shape([lastShape.volume])))
        case let .convolutional( sz , st , n , a):
            precondition(lastShape.dimensions.count == 3, "Convolutions need a three rank Tensor")
            precondition(sz.count == 2 && st.count == 2, "Filter Size and Strides need to be given for x and y direction")
            let sdv = sqrt(2/Float(sz[0] * sz[1] * lastShape[2].value))
            self.layers.append(ConvolutionalLayer(inputShape: lastShape,
                                                  filterAmount: n,
                                                  xFilterSize: sz[0],
                                                  yFilterSize: sz[1],
                                                  xStride: st[0],
                                                  yStride: st[1],
                                                  standardDeviation: sdv,
                                                  activationFunction: a))
        case let .maxPooling( sz, st ):
            precondition(lastShape.dimensions.count == 3, "Max Pooling needs a three rank Tensor")
            precondition(sz.count == 2 && st.count == 2, "Pooling Size and Strides need to be given for x and y direction")
            self.layers.append(MaxPoolingLayer(inputShape: lastShape,
                                               kernelSize: sz + [1],
                                               strides: st + [1]))
        case let .encoder(outputs: o ):
            precondition(lastShape.dimensions.count == 1, "Shape needs to be flattened for an Encoder")
            let sdv = sqrt(2/Float(lastShape[0].value))
            self.layers.append(EncodingLayer(inputs: lastShape[0].value,
                                             outputs: o,
                                             standardDeviation: sdv))
        case let .decoder(encoderIndex: i):
            precondition(self.layers.count > i, "Index needs to be within the amount of added Layers")
            guard let encoder = self.layers[i] as? EncodingLayer else { return }
            self.layers.append(DecodingLayer(encoder: encoder))
        }
        
        self.outputShape = self.layers.last?.outputShape ?? self.inputShape
        
    }
    
    public init(inputShape: Shape, lossFunction: LossFunction) {
        
        self.inputShape = inputShape
        self.inputAmount = 1
        self.outputShape = inputShape
        self.outputAmount = 1
        
        self.aValues = []
        self.zValues = []
        
        self.layers = []
        
        self.lossFunction = lossFunction
        
    }
    
    required public init(from decoder: Decoder) throws {
        
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let lossFunction = try container.decode(LossFunction.self, forKey: .lossFunction)
        let inputAmount = try container.decode(Int.self, forKey: .inputAmount)
        let outputAmount = try container.decode(Int.self, forKey: .outputAmount)
        var layersArrayForType = try container.nestedUnkeyedContainer(forKey: CodingKeys.layers)
        
        var layers = [Layer]()
        
        var layerArray = layersArrayForType
        while(!layersArrayForType.isAtEnd) {
            
            let drink = try layersArrayForType.nestedContainer(keyedBy: LayerTypeKey.self)
            let type = try drink.decode(InternalLayerType.self, forKey: LayerTypeKey.type)
            
            switch type {
            case .activation:
                layers.append(try layerArray.decode(Layer.self))
            case .dense:
                layers.append(try layerArray.decode(DenseLayer.self))
            case .softmax:
                layers.append(try layerArray.decode(SoftmaxLayer.self))
            case .reshape:
                layers.append(try layerArray.decode(ReshapeLayer.self))
            case .convolutional:
                layers.append(try layerArray.decode(ConvolutionalLayer.self))
            case .maxPooling:
                layers.append(try layerArray.decode(MaxPoolingLayer.self))
            case .encoder:
                layers.append(try layerArray.decode(EncodingLayer.self))
            case .decoder:
                let newLayer = try layerArray.decode(DecodingLayer.self)
                newLayer.encoder = layers.first(where: { layer in
                    if let encoder = layer as? EncodingLayer {
                        return encoder.uuid == newLayer.uuid
                    }
                    return false
                }) as? EncodingLayer
                layers.append(newLayer)
            }
            
        }
        
        self.layers = layers
        
        self.inputShape = self.layers.first!.inputShape
        self.inputAmount = inputAmount
        self.outputShape = self.layers.last!.outputShape
        self.outputAmount = outputAmount

        self.aValues = []
        self.zValues = []
        
        self.lossFunction = lossFunction
        
    }
    
    public init(_ network: Network, layers: Int? = nil) {
        
        self.inputShape = network.inputShape
        self.inputAmount = 1
        self.outputShape = network.inputShape
        self.outputAmount = 1
        
        self.aValues = []
        self.zValues = []
        
        if let layersAmount = layers {
            self.layers = Array(network.layers[0..<layersAmount]).map{$0.copy()}
        }else{
            self.layers = network.layers.map{$0.copy()}
        }
        
        self.lossFunction = network.lossFunction
        
    }
    
    internal enum InternalLayerType: String, Codable {
        
        case activation
        case dense
        case softmax
        case reshape
        case convolutional
        case maxPooling
        case encoder
        case decoder
        
    }
    
    private enum CodingKeys: CodingKey {
        
        case lossFunction
        case layers
        case inputAmount
        case outputAmount
        
    }
    
    enum LayerTypeKey: CodingKey {
        case type
    }
    
}

