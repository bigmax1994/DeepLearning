//
//  ConvolutionalLayer.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 16/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation
import Accelerate

final internal class ConvolutionalLayer: Layer {
    
    var filters: Tensor<Float>
    var bias: Tensor<Float>
    
    var aValuesReshaped: [Tensor<Float>]
    
    var strides: [Int]
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        assert(inputs.count == self.inputAmount, "Incorrect Input Shape")
        
        self.zValues = inputs
        self.aValues = self.zValues.map { return self.applyActivationFunction($0) }
        
        self.aValuesReshaped = []
        
        self.newZValues = self.aValues.map { return self.conv3d($0, bias: self.bias) }
        
        return newZValues
        
    }
    
    override func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        let newDerivatives = derivatives.enumerated().map { (i, derivative) in return self.derivativeConv3d(derivative: derivative, i: i) }
        
        return (newDerivatives: newDerivatives.map { return $0.aDerivative }, internalDerivatives: [newDerivatives.map { return $0.filterDerivative }, derivatives].flatMap { return $0 })
        
    }
    
    init(inputShape: Shape, filterAmount: Int, xFilterSize: Int, yFilterSize: Int, xStride: Int, yStride: Int, standardDeviation sdv: Float, activationFunction: ActivationFunction) {
        
        assert(inputShape.dimensions.count == 3, "Convolutional Layer needs to be a 3 Rank Tensor")
        assert(xFilterSize > 0 && yFilterSize > 0, "Convolutional Filters need to be of positive size")
        assert(xStride > 0 && yStride > 0, "Convolutional Strides need to be greater than 0")
        
        let rowStride = yStride
        let colStride = xStride
        
        let outRows = inputShape[0].value.ceilDiv(rowStride)
        let outCols = inputShape[1].value.ceilDiv(colStride)
        
        let outputShape = Shape([Dimension(outRows), Dimension(outCols), Dimension(filterAmount)])
        let weightShape = Shape([Dimension(xFilterSize), Dimension(yFilterSize), inputShape[2], Dimension(filterAmount)])

        self.bias = Tensor<Float>(outputShape, constant: 0)
        self.filters = Tensor<Float>(weightShape, standardDeviation: sdv)
        self.strides = [yStride, xStride, 1]
        self.aValuesReshaped = []
        
        super.init(inputShape: inputShape, outputShape: outputShape, activationFunction: activationFunction)
        
        self.derivativeAmount = 2
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let inputShape = try values.decode(Shape.self, forKey: .inputShape)
        let filters = try values.decode(Tensor<Float>.self, forKey: .filters)
        let biases = try values.decode(Tensor<Float>.self, forKey: .biases)
        let strides = try values.decode([Int].self, forKey: .strides)
        let activationFunction = try values.decode(ActivationFunction.self, forKey: .activationFunction)
        
        let outRows = inputShape[0].value.ceilDiv(strides[1])
        let outCols = inputShape[1].value.ceilDiv(strides[2])
        
        let outputShape = Shape([Dimension(outRows), Dimension(outCols), filters.shape[3]])
        
        self.bias = biases
        self.filters = filters
        self.strides = strides
        self.aValuesReshaped = []
        
        super.init(inputShape: inputShape, outputShape: outputShape, activationFunction: activationFunction)
        
        self.derivativeAmount = 2
        
    }
    
    override func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.inputShape, forKey: .inputShape)
        try container.encode(self.filters, forKey: .filters)
        try container.encode(self.strides, forKey: .strides)
        try container.encode(self.bias, forKey: .biases)
        try container.encode(self.activationFunction, forKey: .activationFunction)
        
        try container.encode(Network.InternalLayerType.convolutional, forKey: .type)

    }
    
    private enum CodingKeys: String, CodingKey {
        
        case inputShape
        case filters
        case strides
        case biases
        case activationFunction
        case type

    }
    
    private func conv3d(_ input: Tensor<Float>, bias: Tensor<Float>) -> Tensor<Float> { // padding = Same
        
        let inChannels = self.filters.shape[2].value
        
        precondition(input.dimensions == 3, "`shape.dimensions.count` must be 3: \(input.dimensions)")
        precondition(self.filters.dimensions == 4, "`filter.shape.dimensions.count` must be 4: \(self.filters.dimensions)")
        precondition(self.strides.count == 3, "`strides.count` must be 3: \(self.strides.count)")
        precondition(self.strides[2] == 1, "`strides[2]` must be 1")
        precondition(input.shape[2].value == inChannels, "The number of channels of this tensor and the filter are not compatible: \(input.shape[2]) != \(inChannels)")
        
        self.aValuesReshaped.append(self.im2col(input))
        
        let reshapedFilter = Tensor<Float>(Shape([Dimension(self.filters.shape[3].value), Dimension(self.filters.shape[0].value * self.filters.shape[1].value * self.filters.shape[2].value)]), values: self.filters.elements)
        
        let resultElements = (reshapedFilter * self.aValuesReshaped[self.aValuesReshaped.count - 1]).fast().elements
        
        return Tensor<Float>(self.outputShape, values: resultElements)
        
    }
    
    private func derivativeConv3d(derivative: Tensor<Float>, i: Int) -> (aDerivative: Tensor<Float>, filterDerivative: Tensor<Float>) {
        
        let reshapedDerivative = Tensor<Float>.reshaped(derivative, to: Shape([Dimension(self.outputShape[0].value * self.outputShape[1].value), Dimension(self.outputShape[2].value)]))
        let reshapedFilter = Tensor<Float>(Shape([Dimension(self.filters.shape[3].value), Dimension(self.filters.shape[0].value * self.filters.shape[1].value * self.filters.shape[2].value)]), values: self.filters.elements)

        var a = (reshapedDerivative * reshapedFilter).fast()
        
        a.elements = self.col2im(a.elements)
        
        let w = (self.aValuesReshaped[i] * reshapedDerivative).fast()
        
        let output = (aDerivative: Tensor<Float>.reshaped(a, to: self.inputShape), filterDerivative: Tensor<Float>.reshaped(w, to: self.filters.shape))
        
        return output
        
    }
    
    private func col2im(_ arr: [Float]) -> [Float] {
        
        let inChannels = self.filters.shape[2].value
        
        let inRows = self.inputShape[0].value
        let inCols = self.inputShape[1].value
        
        let inFaceVolume = inRows * inCols

        let filterHeight = self.filters.shape[0].value
        let filterWidth = self.filters.shape[1].value
        
        let inMinDy = -(filterHeight - 1) / 2
        let inMaxDy = inMinDy + filterHeight - 1
        let inMinDx = -(filterWidth - 1) / 2
        let inMaxDx = inMinDx + filterWidth - 1
        
        let rowStride = strides[0]
        let colStride = strides[1]
        
        let outRows = self.outputShape[0].value
        let outCols = self.outputShape[1].value
        
        let outputArray = [Float](repeating: 0, count: self.inputShape.volume)
        var inputPointer = UnsafePointer<Float>(arr)
        
        for y in 0..<outRows {
            
            let inY0 = y * rowStride
            let inMinY = Swift.max(inY0 + inMinDy, 0)
            let inMaxY = Swift.min(inY0 + inMaxDy, inRows - 1)
                        
            for x in 0..<outCols {
                
                let inX0 = x * colStride
                let inMinX = Swift.max(inX0 + inMinDx, 0)
                let inMaxX = Swift.min(inX0 + inMaxDx, inCols - 1)
                             
                inputPointer -= Swift.min(inY0 + inMinDy, 0) * filterWidth
                inputPointer -= Swift.min(inX0 + inMinDx, 0)

                for i in 0..<inChannels {
                    
                    for j in inMinY...inMaxY {
                                  
                        let outputPointer = UnsafeMutablePointer<Float>(mutating: outputArray) + i * inFaceVolume + j * inCols + inMinX
                        
                        var output:[Float] = []
                        
                        for i in 0..<(inMinX...inMaxX).count {
                            
                            output.append((inputPointer + i).pointee + (outputPointer + i).pointee)
                            
                        }
                        
                        memcpy(outputPointer, &output, MemoryLayout<Float>.size * (inMinX...inMaxX).count)
                        
                        var pointerOffset = filterWidth
                        if j - inMaxY >= 0 { pointerOffset += (filterHeight - inMaxY + inMinY + Swift.min(inY0 + inMinDy, 0) - 1) * filterWidth + Swift.min(inX0 + inMinDx, 0) }
                        
                        inputPointer += pointerOffset
                        
                    }
                    
                }
            
            }
                
        }
                
        return outputArray
        
    }
    
    private func im2col(_ input: Tensor<Float>) -> Tensor<Float> {
        
        let inChannels = self.filters.shape[2].value

        let inRows = self.inputShape[0].value
        let inCols = self.inputShape[1].value
        
        let inFaceVolume = inRows * inCols
        
        let filterHeight = self.filters.shape[0].value
        let filterWidth = self.filters.shape[1].value
        
        let inMinDy = -(filterHeight - 1) / 2
        let inMaxDy = inMinDy + filterHeight - 1
        let inMinDx = -(filterWidth - 1) / 2
        let inMaxDx = inMinDx + filterWidth - 1
        
        let rowStride = self.strides[0]
        let colStride = self.strides[1]
        
        let outRows = self.outputShape[0].value
        let outCols = self.outputShape[1].value
        
        let outFaceVolume = outRows * outCols
        
        let rowSize = filterHeight * filterWidth * inChannels

        let outputArray = [Float](repeating: 0, count: outFaceVolume * rowSize)
        var outputPointer = UnsafeMutablePointer<Float>(mutating: outputArray)
        
        for y in 0..<outRows {
            
            let inY0 = y * rowStride
            let inMinY = Swift.max(inY0 + inMinDy, 0)
            let inMaxY = Swift.min(inY0 + inMaxDy, inRows - 1)
                        
            for x in 0..<outCols {
                
                let inX0 = x * colStride
                let inMinX = Swift.max(inX0 + inMinDx, 0)
                let inMaxX = Swift.min(inX0 + inMaxDx, inCols - 1)
                             
                outputPointer -= Swift.min(inY0 + inMinDy, 0) * filterWidth
                outputPointer -= Swift.min(inX0 + inMinDx, 0)

                for i in 0..<inChannels {
                    
                    for j in inMinY...inMaxY {
                                  
                        let inputPointer = UnsafePointer<Float>(input.elements) + i * inFaceVolume + j * inCols + inMinX
                        
                        memcpy(outputPointer, inputPointer, MemoryLayout<Float>.size * (inMinX...inMaxX).count)
                        
                        var pointerOffset = filterWidth
                        if j - inMaxY >= 0 { pointerOffset += (filterHeight - inMaxY + inMinY + Swift.min(inY0 + inMinDy, 0) - 1) * filterWidth + Swift.min(inX0 + inMinDx, 0) }
                        
                        outputPointer += pointerOffset
                        
                    }
                    
                }
            
            }
                
        }
                
        return Tensor<Float>(Shape([Dimension(rowSize), Dimension(outFaceVolume)]), values: outputArray)
        
    }
 
    override func applyDerivatives(_ derivatives: [Tensor<Float>], learningRate: Float) {
        
        assert(derivatives.count == self.derivativeAmount, "Incorrect amount of Derivatives")
        
        self.filters = Tensor<Float>.reshaped((Tensor<Float>.reshaped(self.filters, to: Shape([self.filters.volume,1])) + (learningRate * Tensor<Float>.reshaped(derivatives[0], to: Shape([derivatives[0].volume,1])))).fast(), to: self.filters.shape)
            self.bias = Tensor<Float>.reshaped((Tensor<Float>.reshaped(self.bias, to: Shape([self.bias.volume,1])) + (learningRate * Tensor<Float>.reshaped(derivatives[1], to: Shape([derivatives[1].volume,1])))).fast(), to: self.bias.shape)

    }
    
    override func copy() -> ConvolutionalLayer {
        
        let newLayer = ConvolutionalLayer(inputShape: self.inputShape, filterAmount: self.filters.shape[3].value, xFilterSize: self.filters.shape[0].value, yFilterSize: self.filters.shape[1].value, xStride: self.strides[0], yStride: self.strides[1], standardDeviation: 0, activationFunction: self.activationFunction)
        newLayer.filters = self.filters
        newLayer.strides = self.strides
        newLayer.bias = self.bias
        
        return newLayer
        
    }
    
}
