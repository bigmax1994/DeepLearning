//
//  MatrixCalc.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation
import Accelerate

public typealias FTensor = Tensor<Float>

public indirect enum TensorCalc {
    
    case t(FTensor)
    case transpose(FTensor)
    case dot(TensorCalc, TensorCalc)
    case sub(TensorCalc, TensorCalc)
    case add(TensorCalc, TensorCalc)
    case constMult(TensorCalc, Float)
    case hadamard(TensorCalc, TensorCalc)

    public func fast() -> FTensor {
        
        switch self {
            
        case let .t(tensor): return tensor
            
        case let .add(.t(a), .t(b)): return a.add(b)
        case let .sub(.t(a), .t(b)): return a.subtract(b)
        case let .hadamard(.t(a), .t(b)): return a.hadamard(b)
            
        case let .constMult(.t(a), value): return a.scale(value)
            
        case let .dot(.t(a), .t(b)): return a.generalDotAdd(alpha: 1, transposeSelf: false, dot: b, transposeDot: false, beta: 0, add: nil)
        case let .dot(.t(a), .transpose(b)): return a.generalDotAdd(alpha: 1, transposeSelf: false, dot: b, transposeDot: true, beta: 0, add: nil)
        case let .dot(.transpose(a), .t(b)): return a.generalDotAdd(alpha: 1, transposeSelf: true, dot: b, transposeDot: false, beta: 0, add: nil)
        case let .dot(.transpose(a), .transpose(b)): return a.generalDotAdd(alpha: 1, transposeSelf: true, dot: b, transposeDot: true, beta: 0, add: nil)
        case let .dot(.constMult(.t(a), value), .t(b)): return a.generalDotAdd(alpha: value, transposeSelf: false, dot: b, transposeDot: false, beta: 0, add: nil)

        case let .add(.dot(.t(a), .t(b)), .t(c)): return a.generalDotAdd(alpha: 1, transposeSelf: false, dot: b, transposeDot: false, beta: 1, add: c)
        case let .add(.t(a), .dot(.t(b), .t(c))): return b.generalDotAdd(alpha: 1, transposeSelf: false, dot: c, transposeDot: false, beta: 1, add: a)
        case let .add(.dot(.t(a), .transpose(b)), .t(c)): return a.generalDotAdd(alpha: 1, transposeSelf: false, dot: b, transposeDot: true, beta: 1, add: c)
        case let .add(.t(a), .dot(.t(b), .transpose(c))): return b.generalDotAdd(alpha: 1, transposeSelf: false, dot: c, transposeDot: true, beta: 1, add: a)
        case let .add(.dot(.transpose(a), .t(b)), .t(c)): return a.generalDotAdd(alpha: 1, transposeSelf: true, dot: b, transposeDot: false, beta: 1, add: c)
        case let .add(.t(a), .dot(.transpose(b), .t(c))): return b.generalDotAdd(alpha: 1, transposeSelf: true, dot: c, transposeDot: false, beta: 1, add: a)
        case let .add(.dot(.transpose(a), .transpose(b)), .t(c)): return a.generalDotAdd(alpha: 1, transposeSelf: true, dot: b, transposeDot: true, beta: 1, add: c)
        case let .add(.t(a), .dot(.transpose(b), .transpose(c))): return b.generalDotAdd(alpha: 1, transposeSelf: true, dot: c, transposeDot: true, beta: 1, add: a)
            
        case let .constMult(.add(.dot(.t(a), .t(b)), .t(c)), value): return a.generalDotAdd(alpha: value, transposeSelf: false, dot: b, transposeDot: false, beta: value, add: c)
            
        case let .add(.constMult(.dot(.t(a), .t(b)), value), .t(c)): return a.generalDotAdd(alpha: value, transposeSelf: false, dot: b, transposeDot: false, beta: 1, add: c)
        case let .add(.dot(.t(a), .t(b)), .constMult(.t(c), value)): return a.generalDotAdd(alpha: value, transposeSelf: false, dot: b, transposeDot: false, beta: value, add: c)
        case let .add(.constMult(.dot(.t(a), .t(b)), valueA), .constMult(.t(c), valueB)): return a.generalDotAdd(alpha: valueA, transposeSelf: false, dot: b, transposeDot: false, beta: valueB, add: c)
            
        case let .constMult(.add(.t(a), .t(b)), value): return a.scalingAdd(b, alpha: value, beta: value)
        case let .add(.constMult(.t(a), value), .t(b)): return a.scalingAdd(b, alpha: value, beta: 1)
        case let .add(.t(a), .constMult(.t(b), value)): return a.scalingAdd(b, alpha: 1, beta: value)
        case let .add(.constMult(.t(a), valueA), .constMult(.t(b), valueB)): return a.scalingAdd(b, alpha: valueA, beta: valueB)
        case let .constMult(.sub(.t(a), .t(b)), value): return a.scalingSubtract(b, alpha: value, beta: value)

        case let .sub(.constMult(.t(a), value), .t(b)): return a.scalingSubtract(b, alpha: value, beta: 1)
        case let .sub(.t(a), .constMult(.t(b), value)): return a.scalingSubtract(b, alpha: 1, beta: value)
        case let .sub(.constMult(.t(a), valueA), .constMult(.t(b), valueB)): return a.scalingSubtract(b, alpha: valueA, beta: valueB)
            
        case let .constMult(.hadamard(.t(a), .t(b)), value): return a.scalingHadamard(b, alpha: value)
        case let .hadamard(.constMult(.t(a), value), .t(b)): return a.scalingHadamard(b, alpha: value)
            
        case let .add(.t(a), .constMult(.transpose(b), value)): return b.generalDotAdd(alpha: value, transposeSelf: true, dot: FTensor.identityMatrix(from: Shape([b.shape[0],b.shape[0]])), transposeDot: false, beta: 1, add: a)

        default: fatalError("Unknown Fast operation")
            
        }
        
    }
    
    public static func +(a: FTensor, b: TensorCalc) -> TensorCalc {
        return .add( .t(a), b )
    }
    
    public static func +(a: TensorCalc, b: FTensor) -> TensorCalc {
        return .add( a, .t(b) )
    }
    
    public static func +(a: TensorCalc, b: TensorCalc) -> TensorCalc {
        return .add( a, b )
    }
    
    public static func *(a: TensorCalc, b: TensorCalc) -> TensorCalc {
        return .dot(a, b)
    }
    
    public static func *(a: TensorCalc, b: FTensor) -> TensorCalc {
        return .dot( a, .t(b))
    }
    
    public static func *(a: FTensor, b: TensorCalc) -> TensorCalc {
        return .dot( .t(a), b)
    }
    
    public static func -(a: TensorCalc, b: FTensor) -> TensorCalc {
        return .sub( a, .t(b))
    }
    
    public static func -(a: FTensor, b: TensorCalc) -> TensorCalc {
        return .sub( .t(a), b)
    }
    
    public static func -(a: TensorCalc, b: TensorCalc) -> TensorCalc {
        return .sub( a, b)
    }
    
    public static func *(a: TensorCalc, b: Float) -> TensorCalc {
        return .constMult( a, b )
    }
    
    public static func *(a: Float, b: TensorCalc) -> TensorCalc {
        return .constMult( b, a )
    }
    
}

extension FTensor {
    
    public func transpose() -> TensorCalc {
        return .transpose(self)
    }
    
    public func len() -> T {
        let sum = self.dot(self)
        return Darwin.sqrt(sum)
    }
    
    public static func *(a: FTensor, b: FTensor) -> TensorCalc {
        return .dot( .t(a), .t(b) )
    }
    
    public static func *(a: FTensor, b: TensorCalc) -> TensorCalc {
        return .dot( .t(a), b )
    }
    
    public static func +(a: FTensor, b: FTensor) -> TensorCalc {
        return .add( .t(a), .t(b) )
    }
    
    public static func -(a: FTensor, b: FTensor) -> TensorCalc {
        return .sub( .t(a), .t(b) )
    }
    
    public static func *(a: FTensor, b: Float) -> TensorCalc {
        return .constMult( .t(a), b )
    }
    
    public static func *(a: Float, b: FTensor) -> TensorCalc {
        return .constMult( .t(b), a )
    }
    
    public static func /(a: FTensor, b: Float) -> TensorCalc {
        return .constMult( .t(a), 1/b )
    }
    
    public static func /(a: Float, b: FTensor) -> TensorCalc {
        return .constMult( .t(b), 1/a )
    }
    
    public static func **(a: FTensor, b: FTensor) -> TensorCalc {
        return .hadamard( .t(a), .t(b) )
    }
    
    public static func /(a: FTensor, b: FTensor) -> TensorCalc {
        var newB = b
        newB.elements = b.elements.map { return 1/$0 }
        return .hadamard( .t(a), .t(newB) )
    }
    
    public static func +(a:FTensor, b: Float) -> FTensor {
        var b = b
        let count = a.volume
        var newValues:[Float] = [Float](repeating: .nan, count: count)
        vDSP_vsadd(a.elements, 1, &b, &newValues, 1, UInt(count))
        return Tensor<Float>(a.shape, values: newValues)
    }
    
    public static func -(a:FTensor, b: Float) -> FTensor {
        var b = -b
        let count = a.volume
        var newValues:[Float] = [Float](repeating: .nan, count: count)
        vDSP_vsadd(a.elements, 1, &b, &newValues, 1, UInt(count))
        return Tensor<Float>(a.shape, values: newValues)
    }
    
}

extension Tensor where T == Float {
    
    public func sin() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(sinf))
    }

    public func cos() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(cosf))
    }

    public func tan() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(tanf))
    }

    public func asin() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(asinf))
    }
    
    public func acos() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(acosf))
    }
    
    public func atan() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(atanf))
    }
    
    public func sinh() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(sinhf))
    }
    
    public func cosh() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(coshf))
    }
    
    public func tanh() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(tanhf))
    }
    
    public func exp() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(expf))
    }
    
    public func log() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(logf))
    }
    
    public func sqrt() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(sqrtf))
    }
    
    public func cbrt() -> Tensor<Float> {
        return Tensor<Float>(shape, values: elements.map(cbrtf))
    }
    
}
