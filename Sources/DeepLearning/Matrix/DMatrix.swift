//
//  DMatrix.swift
//  MachineLearning
//
//  Created by strictlyswift on 9/10/17.
//

import Foundation
import Accelerate
import MetalPerformanceShaders

public typealias DMatrix = Matrix<Float>

public indirect enum MatrixCalc {
    case m(DMatrix)
    case transpose(DMatrix)
    case dot(MatrixCalc, MatrixCalc)
    case sub(MatrixCalc, MatrixCalc)
    case add(MatrixCalc, MatrixCalc)
    case constMult(MatrixCalc, Float)
    case hadamard(MatrixCalc, MatrixCalc)
    
    public func fast() -> DMatrix {
        switch self {
        case let .m(matrix): return matrix
            
        // the cases below have no special leaf-level processing, so use generic case matches
        case let .sub( a, b ) : return a.fast().fastSub( b.fast() )
        case let .constMult( a, value ) : return a.fast().fastMult(value)

        // where we have explicit leaf-level processing, use specific case matches
        // ...for "dot" multiplication
        case let .dot( .m(a), .m(b) ) : return a.dot(b)
        case let .dot( .m(a), .transpose( b) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: false, dot: b, transposeDot: true, beta: 0, add: nil)
        case let .dot( .transpose( a ), .m(b) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: false, beta: 0, add: nil)
        case let .dot( .transpose( a ), .transpose(b) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: true, beta: 0, add: nil)
            
       // ...for addition
        case let .add( .m(a), .m(b) ) : return a.fastAdd(b)
        case let .add( .dot( .m(a), .m(b) ), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: false, dot: b, transposeDot: false, beta: 1.0, add: c)
        case let .add( .dot( .m(a), .transpose(b) ), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: false, dot: b, transposeDot: true, beta: 1.0, add: c)
        case let .add( .dot( .transpose(a), .m(b)), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: false, beta: 1.0, add: c)
        case let .add( .dot( .transpose(a), .transpose(b) ), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: true, beta: 1.0, add: c)
            
        case let .add( .constMult( .dot( .m(a), .m(b)), value), .m(c)) : return a.generalDotAdd(alpha: value, transposeSelf: false, dot: b, transposeDot: false, beta: 1.0, add: c)
        case let .add( .constMult( .dot( .transpose(a), .m(b)), value), .m(c)) : return a.generalDotAdd(alpha: value, transposeSelf: true, dot: b, transposeDot: false, beta: 1.0, add: c)
        case let .add( .constMult( .dot( .m(a), .transpose(b)), value), .m(c)) : return a.generalDotAdd(alpha: value, transposeSelf: false, dot: b, transposeDot: true, beta: 1.0, add: c)
        case let .add( .constMult( .dot( .transpose(a), .transpose(b)), value), .m(c)) : return a.generalDotAdd(alpha: value, transposeSelf: true, dot: b, transposeDot: true, beta: 1.0, add: c)
            
        //case let .add( .constMult( .m(a), value), .m(c)) : return c.fastAdd(a.fastMult(value))
        case let .add( .constMult( .transpose(a), value), .m(c)) : return c.fastAdd(a.T().fastMult(value))
        case let .add( .constMult( .m(a), value), .transpose(c)) : return c.T().fastAdd(a.fastMult(value))
        case let .add( .constMult( .transpose(a), value), .transpose(c)) : return c.T().fastAdd(a.T().fastMult(value))
            
        case let .add( .dot( .m(a), .m(b) ), .add( .dot( .m(c), .m(d) ), .m(e)) ) : return a.generalDotAdd(alpha: 1, transposeSelf: false, dot: b, transposeDot: false, beta: 1, add: c.generalDotAdd(alpha: 1, transposeSelf: false, dot: d, transposeDot: false, beta: 1, add: e))
            
        case let .dot( .hadamard( .m(a), .m(b) ), .m(c) ): return c.dot(a.fastHadamard(b))
            
        case let .add( .constMult( .m(a), value ), .m(b) ): return a.scalingAdd(alpha: value, b: b, beta: 1)
        case let .add( .m(a), .constMult( .m(b), value ) ): return a.scalingAdd(alpha: 1, b: b, beta: value)
        case let .add( .constMult( .m(a), valueA ), .constMult( .m(b), valueB ) ): return a.scalingAdd(alpha: valueA, b: b, beta: valueB)
            
        case let .hadamard( .m(a), .m(b) ): return a.fastHadamard(b)
            
        default: fatalError("Could not calculate matrix")
        }
    }
    
    public static func +(a: MatrixCalc, b: DMatrix) -> MatrixCalc {
        return .add( a, .m(b) )
    }
    
    public static func +(a: MatrixCalc, b: MatrixCalc) -> MatrixCalc {
        return .add( a, b )
    }
    
    public static func ●(a: MatrixCalc, b: MatrixCalc) -> MatrixCalc {
        return .dot(a, b)
    }
    
    public static func ●(a: MatrixCalc, b: DMatrix) -> MatrixCalc {
        return .dot( a, .m(b)) 
    }
    
    public static func -(a: MatrixCalc, b: DMatrix) -> MatrixCalc {
        return .sub( a, .m(b))
    }
    
    public static func -(a: DMatrix, b: MatrixCalc) -> MatrixCalc {
        return .sub( .m(a), b)
    }
    
    public static func *(a: MatrixCalc, b: Float) -> MatrixCalc {
        return .constMult( a, b )
    }
    
    public static func *(a: Float, b: MatrixCalc) -> MatrixCalc {
        return .constMult( b, a )
    }
}

extension Matrix where T == Float {

    public init(_ rows: Int, _ cols: Int, min: T, max: T ) {
        self.rows = rows
        self.cols = cols
        
        self.values = Array<T>(repeating: 0.0, count: rows*cols)
        
        for i in 0..<values.count {
            
            self.values[i] = Float.random(in: min...max)
            
        }
    }

    
    public func printFormatted() -> Void {
        var s = "< "
        for row in 0..<rows {
            for col in 0..<cols {
                let val = self[row,col]
                print( String(format:"%@%.2f ", (val<0 ? "" : " "),val), terminator: "" , to: &s)
            }
            print(s)
            s = "  "
        }
        print(">")
    }
    
    func dot(_ other: Matrix<Float>) -> Matrix<Float> {
        return generalDotAdd(alpha: 1.0, transposeSelf: false, dot: other, transposeDot: false, beta: 0.0, add: nil)
    }
    
    func dotAdd(dot: Matrix<Float>, add:Matrix<Float>) -> Matrix<Float> {
        return generalDotAdd(alpha: 1.0, transposeSelf: false, dot: dot, transposeDot: false, beta: 1.0, add: add)
    }
    
    /// Calculates  alpha(self ● dot) + beta(add).   Either self or dot may be transposed
    func generalDotAdd(alpha: Float, transposeSelf: Bool, dot: Matrix<Float>, transposeDot: Bool, beta: Float, add:Matrix<Float>?) -> Matrix<Float> {
        
        let operations = Float(self.rows * dot.cols * (self.cols + (self.cols - 1)))
        var m = Matrix<Float>(self.rows, dot.cols, constant: 0.0)
        
        if operations /*< pow(10, 9)*/ > -1 {
            
            let selfAngle = (rows:(transposeSelf ? self.cols : self.rows), cols:(transposeSelf ? self.rows : self.cols))
            let dotAngle = (rows:(transposeDot ? dot.cols : dot.rows), cols:(transposeDot ? dot.rows: dot.cols ))
            assert( selfAngle.cols == dotAngle.rows, "Matrices are not compatible for multiplication")
            
            if add != nil {
                assert( selfAngle.rows == add!.rows && dotAngle.cols == add!.cols, "Matrices are not compatible for addition")
                m.values = add!.values
            }
            
            cblas_sgemm(CblasRowMajor, transposeSelf ? CblasTrans : CblasNoTrans, transposeDot ? CblasTrans : CblasNoTrans, Int32(selfAngle.rows), Int32(dotAngle.cols), Int32(selfAngle.cols), alpha, self.values, Int32(self.cols), dot.values, Int32(dot.cols), beta, &m.values, Int32(m.cols))
            
        }else{
            
            /*if let device = MTLCreateSystemDefaultDevice() {
                
                if let commandQueue = device.makeCommandQueue() {
                    
                    if add != nil {
                                            
                        if let bufferA = device.makeBuffer(bytes: self.values, length: self.rows * self.cols * MemoryLayout<Float>.stride, options: []) {
                            
                            let descA = MPSMatrixDescriptor(dimensions: self.rows, columns: self.cols, rowBytes: self.cols * MemoryLayout<Float>.stride, dataType: .float32)
                        
                            let MatrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
                            
                            if let bufferB = device.makeBuffer(bytes: dot.values, length: dot.rows * dot.cols * MemoryLayout<Float>.stride, options: []) {
                                
                                let descB = MPSMatrixDescriptor(dimensions: dot.rows, columns: dot.cols, rowBytes: dot.cols * MemoryLayout<Float>.stride, dataType: .float32)
                            
                                let MatrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
                                
                                if let bufferC = device.makeBuffer(bytes: add!.values, length: add!.rows * add!.cols * MemoryLayout<Float>.stride, options: []) {
                                    
                                    let descC = MPSMatrixDescriptor(dimensions: add!.rows, columns: add!.cols, rowBytes: add!.cols * MemoryLayout<Float>.stride, dataType: .float32)
                                
                                    let MatrixC = MPSMatrix(buffer: bufferC, descriptor: descC)
                                    
                                    let matrixMultiplication = MPSMatrixMultiplication(device: device, transposeLeft: transposeSelf, transposeRight: transposeDot, resultRows: add!.rows, resultColumns: add!.cols, interiorColumns: self.cols, alpha: Double(alpha), beta: Double(beta))
                                    
                                    if let commandBuffer = commandQueue.makeCommandBuffer() {
                                    
                                        matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: MatrixA, rightMatrix: MatrixB, resultMatrix: MatrixC)
                                        
                                        commandBuffer.commit()
                                        
                                        let rawPointer = MatrixC.data.contents()
                                        let count = MatrixC.rows * MatrixC.columns
                                        let typedPointer = rawPointer.bindMemory(to: Float.self, capacity: count)
                                        let bufferedPointer = UnsafeBufferPointer(start: typedPointer, count: count)
                                        
                                        let resultArray = [Float](bufferedPointer)
                                        
                                        m.values = resultArray
                                        
                                    }
                                    
                                }
                                
                            }
                            
                        }
                        
                    }
                    
                }
                
            }*/
            
        }
        
        return m
        
    }
    
    func fastApply(other: DMatrix, with f: (Float, Float) -> Float, name: String) -> DMatrix {
        assert( self.rows == other.rows && self.cols == other.cols, "Matrices are not compatible for \(name)")
        var m = Matrix<Float>( self.rows, self.cols, constant: 0.0)
        for i in 0..<self.values.count {
            m.values[i] = f( self.values[i], other.values[i] )
        }
        
        return m
    }
    
    /// calculates self + b  quickly by avoiding the memory overhead of the zip
    func fastAdd(_ b: DMatrix) -> DMatrix { //return self.fastApply(other: b, with: (+), name: "addition")
        
        assert( self.rows == b.rows && self.cols == b.cols, "Matrices are not compatible for Addition")
        
        /*var m = Matrix<Float>(self.rows, b.cols, constant: 0.0)

        if let device = MTLCreateSystemDefaultDevice() {
            
            if let commandQueue = device.makeCommandQueue() {
                                
                 if let bufferA = device.makeBuffer(bytes: self.values, length: self.rows * self.cols * MemoryLayout<Float>.stride, options: []) {
                                           
                    let descA = MPSMatrixDescriptor(dimensions: self.rows, columns: self.cols, rowBytes: self.cols * MemoryLayout<Float>.stride, dataType: .float32)
                                       
                    let MatrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
                                           
                    if let bufferB = device.makeBuffer(bytes: b.values, length: b.rows * b.cols * MemoryLayout<Float>.stride, options: []) {
                                               
                        let descB = MPSMatrixDescriptor(dimensions: b.rows, columns: b.cols, rowBytes: b.cols * MemoryLayout<Float>.stride, dataType: .float32)
                                           
                        let MatrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
                                               
                        if let bufferC = device.makeBuffer(bytes: m.values, length: m.rows * m.cols * MemoryLayout<Float>.stride, options: []) {
                            
                            let descC = MPSMatrixDescriptor(dimensions: m.rows, columns: m.cols, rowBytes: m.cols * MemoryLayout<Float>.stride, dataType: .float32)
                            
                            let MatrixC = MPSMatrix(buffer: bufferC, descriptor: descC)
                            
                            let matrixAddition = MPSMatrixSum(device: device, count: 2, rows: self.rows, columns: self.cols, transpose: false)
                            
                            if let commandBuffer = commandQueue.makeCommandBuffer() {
                                
                                matrixAddition.encode(to: commandBuffer, sourceMatrices: [MatrixA,MatrixB], resultMatrix: MatrixC, scale: nil, offsetVector: nil, biasVector: nil, start: 0)
                                
                                commandBuffer.commit()
                                
                                let rawPointer = MatrixC.data.contents()
                                let count = MatrixC.rows * MatrixC.columns
                                let typedPointer = rawPointer.bindMemory(to: Float.self, capacity: count)
                                let bufferedPointer = UnsafeBufferPointer(start: typedPointer, count: count)
                                
                                let resultArray = [Float](bufferedPointer)
                                
                                m.values = resultArray
                                                                
                            }
                            
                        }
                        
                    }
                    
                }
                
            }
            
        }
        
        return m*/
        
        let identityMatrix = Matrix<Float>.I(self.cols)
        
        return self.generalDotAdd(alpha: 1, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: 1, add: b)
        
    }
    
    func scalingAdd(alpha: Float, b: DMatrix, beta: Float) -> DMatrix {
        
        let identityMatrix = Matrix<Float>.I(self.cols)
        
        return self.generalDotAdd(alpha: alpha, transposeSelf: false, dot: identityMatrix, transposeDot: false, beta: beta, add: b)
        
    }
    
    /// calculates self - b  quickly by avoiding the memory overhead of the zip
    func fastSub(_ b: DMatrix) -> DMatrix { return self.fastApply(other: b, with: (-), name: "subtraction") }

    /// calculates self ⊙ b  quickly by avoiding the memory overhead of the zip
    func fastHadamard(_ b: DMatrix) -> DMatrix { return self.fastApply(other: b, with: (*), name: "Hadamard") }

    mutating func inplaceAdd(_ other: DMatrix) {
        assert( self.rows == other.rows && self.cols == other.cols, "Matrices are not compatible for addition")
        for i in 0..<self.values.count {
            self.values[i] = self.values[i] + other.values[i]
        }
    }
    
    func transpose() -> MatrixCalc {
        return .transpose( self)
    }

    /// calculates self * value  quickly using BLAS
    func fastMult(_ b: Float) -> DMatrix {
        var m = Matrix<Float>( self.rows, self.cols, constant: 0.0)
        m.values = self.values

        cblas_sscal(Int32(self.rows * self.cols), b, &m.values, 1)
        return m
    }
    
    public static func ●(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .dot( .m(a), .m(b) )
    }
    
    public static func ●(a: DMatrix, b: MatrixCalc) -> MatrixCalc {
        return .dot( .m(a), b )
    }
    
    public static func +(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .add( .m(a), .m(b) )
    }
    
    public static func -(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .sub( .m(a), .m(b) )
    }
    
    public static func *(a: DMatrix, b: Float) -> MatrixCalc {
        return .constMult( .m(a), b )
    }
    
    public static func *(a: Float, b: DMatrix) -> MatrixCalc {
        return .constMult( .m(b), a )
    }
    
    public static func ⊙(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .hadamard( .m(a), .m(b) )
    }
    
    public static func +=(a: inout DMatrix, b: DMatrix) -> Void {   // should really be MatrixCalc
        return a.inplaceAdd(b)
    }
}
