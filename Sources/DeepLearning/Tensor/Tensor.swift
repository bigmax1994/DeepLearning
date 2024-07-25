//
//  Tensor.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation
import Accelerate

infix operator **: MultiplicationPrecedence

public struct Tensor<T: Codable>: Codable {
    
    var shape: Shape
    var dimensions: Int {
        get {
            return self.shape.dimensions.count
        }
    }
    var volume: Int {
        get {
            return self.shape.volume
        }
    }
    public var elements: [T]
    
    public init(_ shape: Shape, constant: T) {
        
        self.shape = shape
        
        self.elements = Array(repeating: constant, count: self.shape.volume)
        
    }
    
    public init(_ shape: Shape, values: [T]) {
        
        assert(shape.volume == values.count, "Incorrect Shape")
        
        self.shape = shape
        self.elements = values
        
    }
        
    private func checkCompatabilityForDotProduct(_ otherShape: Shape) -> Bool {
        
        return self.shape[1] == otherShape[0] || self.shape[0] == otherShape[1]
        
    }
    
    internal mutating func reshape(to newShape: Shape) {
        
        assert(newShape.volume == self.shape.volume, "Could not reshape Tensor")
        
        self.shape = newShape
        
    }
    
    internal static func reshaped(_ tensor: Tensor<T>, to newShape: Shape) -> Tensor<T> {
        
        assert(newShape.volume == tensor.elements.count, "Could not reshape Tensor")
        
        return Tensor<T>(newShape, values: tensor.elements)
        
    }
    
    public subscript (_ indices: Int...) -> T {
        get {
            return self.elements[getIndex(from: indices)]
        }
        set {
            self.elements[getIndex(from: indices)] = newValue
        }
    }
    
    private func getIndex(from indices: [Int]) -> Int {
        
        assert(indices.count == self.dimensions, "Incorrect amount of indecies")
        
        let amountOfIndices = indices.count - 1
        
        var finalIndex = 0
        
        for i in 0...amountOfIndices {
            
            var value = indices[i]
            
            if i < amountOfIndices {
                
                for j in i..<amountOfIndices {
                    
                    value *= self.shape[j].value
                    
                }
                
            }
            
            finalIndex += value
            
        }
        
        return finalIndex
        
    }
    
}

extension Tensor: Equatable where T: Equatable {
    
    public static func == (lhs: Tensor<T>, rhs: Tensor<T>) -> Bool {
        
        return lhs.shape == rhs.shape && lhs.elements == rhs.elements
        
    }
    
}

extension Tensor where T == Float {
    
    internal init(_ shape: Shape, standardDeviation sdv: Float) {
        
        self.shape = shape
        
        let randomizer = GaussianDistribution(mean: 0, deviation: sdv)
        
        self.elements = Array(repeating: T.zero, count: self.shape.volume)
        
        for i in self.elements.indices {
            
            self.elements[i] = randomizer.nextFloat()
            
        }
        
    }
    
    internal static func identityMatrix(from shape: Shape) -> Tensor<T> {
        
        assert(shape.dimensions.count == 2, "Cannot create Identity Matrix with this Shape")
        assert(shape[0] == shape[1], "Can only create square Identity Matrix")
        
        var newMatrix = Tensor<T>(shape, constant: 0)
        
        for i in 0..<newMatrix.shape[0].value {
            
            newMatrix[i, i] = 1
            
        }
        
        return newMatrix
        
    }
    
    public init(_ shape: Shape, range: ClosedRange<T>) {
        
        self.shape = shape
        
        self.elements = Array(repeating: T.zero, count: self.shape.volume)
        
        for i in self.elements.indices {
            
            self.elements[i] = T.random(in: range)
            
        }
        
    }
    
    public func findMax() -> Float {
        
        return vDSP.maximum(self.elements)
        
        /*var max = -T.infinity
        
        for j in self.elements {
            
            if j > max {
                
                max = j
                
            }
            
        }
        
        return max*/
        
    }
    
    public func sum() -> Float {
        
        return vDSP.sum(self.elements)
        
    }
    
    internal func generalDotAdd(alpha: Float, transposeSelf: Bool, dot: Tensor<T>, transposeDot: Bool, beta: Float, add:Tensor<T>?) -> Tensor<T> {
        
        if self.dimensions == 1 && dot.dimensions == 1 && (transposeDot != transposeSelf) {
            
            var m = Tensor<T>(Shape([1]), constant: 0)
            
            precondition(self.shape[0] == dot.shape[0], "Vectors are not compatible for dot-product")
            
            if add != nil {
                precondition(add!.shape[0].value == 1, "Matrices are not compatible for addition")
                m.elements = add!.elements
            }
            
            cblas_sgemm(CblasRowMajor,
                        transposeSelf ? CblasTrans : CblasNoTrans,
                        transposeDot ? CblasTrans : CblasNoTrans,
                        Int32(1),
                        Int32(1),
                        Int32(self.shape[0].value),
                        alpha,
                        self.elements,
                        Int32(1),
                        dot.elements,
                        Int32(dot.shape[0].value),
                        beta,
                        &m.elements,
                        Int32(m.shape[0].value))
            
            return m
            
        } else if self.dimensions == 1 && dot.dimensions == 1 {
            
            var m = Tensor<T>(Shape([self.shape[0], dot.shape[0]]), constant: 0)
            
            let selfAngle = (rows:(transposeSelf ? 1 : self.shape[0].value),
                             cols:(transposeSelf ? self.shape[0].value : 1))
            let dotAngle = (rows:(transposeDot ? dot.shape[0].value : 1),
                            cols:(transposeDot ? 1 : dot.shape[0].value))
            
            precondition(selfAngle.cols == dotAngle.rows, "Matrices are not compatible for multiplication")
            
            if add != nil {
                precondition(selfAngle.rows == add!.shape[0].value && dotAngle.cols == add!.shape[1].value, "Matrices are not compatible for addition")
                m.elements = add!.elements
            }
            
            cblas_sgemm(CblasRowMajor,
                        transposeSelf ? CblasTrans : CblasNoTrans,
                        transposeDot ? CblasTrans : CblasNoTrans,
                        Int32(selfAngle.rows),
                        Int32(dotAngle.cols),
                        Int32(selfAngle.cols),
                        alpha,
                        self.elements,
                        Int32(1),
                        dot.elements,
                        Int32(dot.shape[0].value),
                        beta,
                        &m.elements,
                        Int32(m.shape[1].value))
            
            return m
            
        }
        
        precondition(self.dimensions <= 2 && dot.dimensions <= 2, "Can only multiply Matricies and Vectors")
                
        if self.dimensions == 1 {
            
            
            let selfAngle = (rows:(transposeSelf ? 1 : self.shape[0].value),
                             cols:(transposeSelf ? self.shape[0].value : 1))
            let dotAngle = (rows:(transposeDot ? dot.shape[1].value : dot.shape[0].value),
                            cols:(transposeDot ? dot.shape[0].value : dot.shape[1].value ))
            
            var m = Tensor<T>(Shape([Dimension(selfAngle.rows), Dimension(dotAngle.cols)]), constant: 0)
            
            precondition(selfAngle.cols == dotAngle.rows, "Matrices are not compatible for multiplication")
            
            if add != nil {
                
                if add!.dimensions == 2 {
                
                    precondition( selfAngle.rows == add!.shape[0].value && dotAngle.cols == add!.shape[1].value, "Matrices are not compatible for addition")
                    m.elements = add!.elements
                    
                }else if add!.dimensions == 1 {
                    
                    precondition( selfAngle.rows == add!.shape[0].value && dotAngle.cols == 1, "Matrices are not compatible for addition")
                    m.elements = add!.elements
                    
                }
                
            }
            
            cblas_sgemm(CblasRowMajor,
                        transposeSelf ? CblasTrans : CblasNoTrans,
                        transposeDot ? CblasTrans : CblasNoTrans,
                        Int32(selfAngle.rows),
                        Int32(dotAngle.cols),
                        Int32(selfAngle.cols),
                        alpha,
                        self.elements,
                        1,
                        dot.elements,
                        Int32(dot.shape[1].value),
                        beta,
                        &m.elements,
                        Int32(m.shape[1].value))
            
            return Tensor<Float>.reshaped(m, to: Shape([m.shape[0]]))
            
        }else if dot.dimensions == 1 {
            
            let selfAngle = (rows:(transposeSelf ? self.shape[1].value : self.shape[0].value),
                             cols:(transposeSelf ? self.shape[0].value : self.shape[1].value))
            let dotAngle = (rows:(transposeDot ? 1 : dot.shape[0].value),
                            cols:(transposeDot ? dot.shape[0].value : 1 ))
            
            var m = Tensor<T>(Shape([Dimension(selfAngle.rows), Dimension(dotAngle.cols)]), constant: 0)
            
            precondition(selfAngle.cols == dotAngle.rows, "Matrices are not compatible for multiplication")
            
            if add != nil {
                
                if add!.dimensions == 2 {
                
                    precondition( selfAngle.rows == add!.shape[0].value && dotAngle.cols == add!.shape[1].value, "Matrices are not compatible for addition")
                    m.elements = add!.elements
                    
                }else if add!.dimensions == 1 {
                    
                    precondition( selfAngle.rows == add!.shape[0].value && dotAngle.cols == 1, "Matrices are not compatible for addition")
                    m.elements = add!.elements
                    
                }
                
            }
            
            cblas_sgemm(CblasRowMajor,
                        transposeSelf ? CblasTrans : CblasNoTrans,
                        transposeDot ? CblasTrans : CblasNoTrans,
                        Int32(selfAngle.rows),
                        Int32(dotAngle.cols),
                        Int32(selfAngle.cols),
                        alpha,
                        self.elements,
                        Int32(self.shape[1].value),
                        dot.elements,
                        Int32(1),
                        beta,
                        &m.elements,
                        Int32(m.shape[1].value))
            
            return Tensor<Float>.reshaped(m, to: Shape([m.shape[0]]))
            
        }
        
        let selfAngle = (rows:(transposeSelf ? self.shape[1].value : self.shape[0].value),
                         cols:(transposeSelf ? self.shape[0].value : self.shape[1].value))
        let dotAngle = (rows:(transposeDot ? dot.shape[1].value : dot.shape[0].value),
                        cols:(transposeDot ? dot.shape[0].value : dot.shape[1].value ))
        
        var m = Tensor<T>(Shape([Dimension(selfAngle.rows), Dimension(dotAngle.cols)]), constant: 0)
        
        precondition(selfAngle.cols == dotAngle.rows, "Matrices are not compatible for multiplication")
        
        if add != nil {
            precondition( selfAngle.rows == add!.shape[0].value && dotAngle.cols == add!.shape[1].value, "Matrices are not compatible for addition")
            m.elements = add!.elements
        }
        
        cblas_sgemm(CblasRowMajor,
                    transposeSelf ? CblasTrans : CblasNoTrans,
                    transposeDot ? CblasTrans : CblasNoTrans,
                    Int32(selfAngle.rows),
                    Int32(dotAngle.cols),
                    Int32(selfAngle.cols),
                    alpha,
                    self.elements,
                    Int32(self.shape[1].value),
                    dot.elements,
                    Int32(dot.shape[1].value),
                    beta,
                    &m.elements,
                    Int32(m.shape[1].value))
        
        return m
        
    }

    internal func add(_ b: Tensor<T>) -> Tensor<T> {
        
        assert(self.shape == b.shape, "Matrices are not compatible for addition")
        
        if self.dimensions == 2 {
        
            let identityMatrix = Tensor<T>.identityMatrix(from: Shape([self.shape[1],self.shape[1]]))
        
            return self.generalDotAdd(alpha: 1, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: 1, add: b)
        
        }else{
            
            let identityMatrix = Tensor<T>.identityMatrix(from: Shape([self.shape[0],self.shape[0]]))
            
            return identityMatrix.generalDotAdd(alpha: 1, transposeSelf: false, dot: self, transposeDot: false, beta: 1, add: b)
            
        }
        
    }
    
    internal func scalingAdd(_ b: Tensor<T>, alpha: T, beta: T) -> Tensor<T> {
        
        assert(self.shape == b.shape, "Matrices are not compatible for addition")
        
        if self.shape.dimensions.count == 2 {
        
            let identityMatrix = Tensor<T>.identityMatrix(from: Shape([self.shape[1],self.shape[1]]))
        
            return self.generalDotAdd(alpha: alpha, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: beta, add: b)
        
        }else{
            
            let identityMatrix = Tensor<T>.identityMatrix(from: Shape([1,1]))
            
            return self.generalDotAdd(alpha: alpha, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: beta, add: b)
            
        }
        
    }
    
    internal func subtract(_ b: Tensor<T>) -> Tensor<T> {
        
        assert(self.shape == b.shape, "Matrices are not compatible for addition")
        
        let identityMatrix = Tensor<T>.identityMatrix(from: self.shape)
        
        var newB = b
        newB.elements = newB.elements.map { -$0 }
        
        return self.generalDotAdd(alpha: 1, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: 1, add: newB)
        
    }
    
    internal func scalingSubtract(_ b: Tensor<T>, alpha: T, beta: T) -> Tensor<T> {
        
        assert(self.shape == b.shape, "Matrices are not compatible for addition")
        
        let identityMatrix = Tensor<T>.identityMatrix(from: self.shape)
        
        var newB = b
        newB.elements = newB.elements.map { -$0 }
        
        return self.generalDotAdd(alpha: alpha, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: beta, add: newB)
        
    }
    
    internal func hadamard(_ b: Tensor<T>) -> Tensor<T> {
        
        assert(self.shape == b.shape)
        
        var newB = b
        newB.elements = newB.elements.enumerated().map { (i, element) in
            
            return element * self.elements[i]
            
        }
        
        return newB
        
    }
    
    internal func scalingHadamard(_ b: Tensor<T>, alpha: T) -> Tensor<T> {
        
        assert(self.shape == b.shape)
        
        var newB = b
        newB.elements = newB.elements.enumerated().map { (i, element) in
            
            return alpha * element * self.elements[i]
            
        }
        
        return newB
        
    }
    
    internal func scale(_ alpha: T) -> Tensor<T> {
        
        var m = self
        
        cblas_sscal(Int32(m.elements.count), alpha, &m.elements, 1)
        
        return m
        
    }
    
    public func dot(_ dot: Tensor<Float>) -> Float {
        
        precondition(self.shape.dimensions.count == 1, "can only dot Vectors")
        precondition(dot.shape.dimensions.count == 1, "can only dot Vectors")
        
        precondition(self.shape.volume == dot.shape.volume, "can only dot Vectors of same length")
        
        //let dotTest = targetEmbedding.elements.enumerated().reduce(0) {$0 + ($1.element * contextEmbedding[$1.offset])}
        
        var dotValue:Float = 0
        
        for i in 0..<self.shape.volume {
            
            dotValue += self[i] * dot[i]
            
        }
        
        return dotValue
        
    }
    
}
