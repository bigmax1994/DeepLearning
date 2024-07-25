//
//  MaxPoolLayer.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 16/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

final internal class MaxPoolingLayer: Layer {
    
    private var kernelSize: [Int]
    private var strides: [Int]
    
    private var indicies: [[Int:Int]] = []
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        self.zValues = inputs
        self.aValues = inputs.map { return self.maxPool($0) }
        self.newZValues = self.aValues
        
        return aValues
        
    }
    
    override func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        return (newDerivatives: derivatives.enumerated().map { (i, derivative) in self.derivativeMaxPool(derivative, i: i) }, internalDerivatives: [])
        
    }
    
    init(inputShape: Shape, kernelSize: [Int], strides: [Int]) {
        
        precondition(inputShape.dimensions.count == 3, "`shape.dimensions.count` must be 3: \(inputShape.dimensions.count)")
        precondition(kernelSize.count == 3, "`ksize.count` must be 3: \(kernelSize.count)")
        precondition(kernelSize[2] == 1, "`ksize[3]` != 1 is not supported: \(kernelSize[2])")
        precondition(strides.count == 3, "`strides.count` must be 3: \(strides.count)")
        precondition(strides[2] == 1, "`strides[2]` != 1 is not supported: \(strides[2])")
        
        self.kernelSize = kernelSize
        self.strides = strides
        
        let rowStride = self.strides[0]
        let colStride = self.strides[1]
        
        let outRows = inputShape[0].value.ceilDiv(rowStride)
        let outCols = inputShape[1].value.ceilDiv(colStride)
        let numChannels = inputShape[2].value

        super.init(inputShape: inputShape, outputShape: Shape([Dimension(outRows), Dimension(outCols), Dimension(numChannels)]), activationFunction: .normal)
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let inputShape = try values.decode(Shape.self, forKey: .inputShape)
        let kernelSize = try values.decode([Int].self, forKey: .kernelSize)
        let strides = try values.decode([Int].self, forKey: .strides)
        
        self.kernelSize = kernelSize
        self.strides = strides
        
        let rowStride = self.strides[0]
        let colStride = self.strides[1]
        
        let outRows = inputShape[0].value.ceilDiv(rowStride)
        let outCols = inputShape[1].value.ceilDiv(colStride)
        let numChannels = inputShape[2].value

        super.init(inputShape: inputShape, outputShape: Shape([Dimension(outRows), Dimension(outCols), Dimension(numChannels)]), activationFunction: .normal)
        
    }
    
    override func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.inputShape, forKey: .inputShape)
        try container.encode(self.kernelSize, forKey: .kernelSize)
        try container.encode(self.strides, forKey: .strides)

        try container.encode(Network.InternalLayerType.maxPooling, forKey: .type)

    }

    private enum CodingKeys: String, CodingKey {
        
        case inputShape
        case kernelSize
        case strides
        case type

    }
    
    private func maxPool(_ input: Tensor<Float>) -> Tensor<Float> { // padding = Same
        
        precondition(self.inputShape.dimensions.count == 3, "`shape.dimensions.count` must be 3: \(self.inputShape.dimensions.count)")
        precondition(self.kernelSize.count == 3, "`ksize.count` must be 3: \(self.kernelSize.count)")
        precondition(self.kernelSize[2] == 1, "`ksize[3]` != 1 is not supported: \(self.kernelSize[2])")
        precondition(self.strides.count == 3, "`strides.count` must be 3: \(self.strides.count)")
        precondition(self.strides[2] == 1, "`strides[2]` != 1 is not supported: \(self.strides[2])")
        
        let inRows = self.inputShape[0].value
        let inCols = self.inputShape[1].value
        let numChannels = self.inputShape[2].value
        
        let filterHeight = self.kernelSize[0]
        let filterWidth = self.kernelSize[1]
        
        let inMinDy = -(filterHeight - 1) / 2
        let inMaxDy = inMinDy + filterHeight - 1
        let inMinDx = -(filterWidth - 1) / 2
        let inMaxDx = inMinDx + filterWidth - 1
        
        let rowStride = self.strides[0]
        let colStride = self.strides[1]

        let outRows = self.inputShape[0].value.ceilDiv(rowStride)
        let outCols = self.inputShape[1].value.ceilDiv(colStride)
        
        var newIndex:[Int:Int] = [:]
        
        // Initialize with -infinity for maximization.
        let elements = [Float](repeating: -Float.infinity, count: outCols * outRows * numChannels)
        
        for y in 0..<outRows {
            let inY0 = y * rowStride
            let inMinY = Swift.max(inY0 + inMinDy, 0)
            let inMaxY = Swift.min(inY0 + inMaxDy, inRows - 1)

            for inY in inMinY...inMaxY {
                var outPixelIndex = y * outCols
                for x in 0..<outCols {
                                        
                    let inX0 = x * colStride
                    let inMinX = Swift.max(inX0 + inMinDx, 0)
                    let inMaxX = Swift.min(inX0 + inMaxDx, inCols - 1)
                    
                    var inPointer = UnsafeMutablePointer<Float>(mutating: input.elements) + (inY * inCols + inMinX) * numChannels
                    for _ in inMinX...inMaxX {
                        var outPointer = UnsafeMutablePointer<Float>(mutating: elements) + outPixelIndex * numChannels
                        
                        if let _ = newIndex[outPixelIndex] {} else {
                        
                            newIndex.updateValue(outPointer - UnsafeMutablePointer<Float>(mutating: elements), forKey: outPixelIndex)
                        
                        }
                        
                        for _ in 0..<numChannels {
                            
                            if outPointer.pointee < inPointer.pointee {
                            
                                outPointer.pointee = inPointer.pointee
                                newIndex.updateValue(inPointer - UnsafeMutablePointer<Float>(mutating: input.elements), forKey: outPixelIndex)
                                
                            }
                            
                            outPointer += 1
                            inPointer += 1
                        }
                    }
                    outPixelIndex += 1
                }
            }
        }
        
        self.indicies.append(newIndex)
        
        return Tensor(self.outputShape, values: elements)
        
    }
    
    private func derivativeMaxPool(_ derivative: Tensor<Float>, i: Int) -> Tensor<Float> {
        
        precondition(derivative.shape == self.outputShape, "Derivative is of incorrect Shape")
        
        var newDerivative = Tensor<Float>(self.inputShape, constant: 0)
        
        for j in 0..<derivative.elements.count {
            
            if let index = self.indicies[i][j] {
                
                newDerivative.elements[index] = derivative.elements[j]
                
            }
            
        }
        
        return newDerivative
        
    }
    
    override func copy() -> MaxPoolingLayer {
        
        let newLayer = MaxPoolingLayer(inputShape: self.inputShape, kernelSize: self.kernelSize, strides: self.strides)
        
        return newLayer
        
    }
    
}
