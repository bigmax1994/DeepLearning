//
//  Main.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 12/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

internal protocol Differentiable {
    
    var inputShape:Shape { get }
    var inputAmount:Int { get }
    
    var outputShape:Shape { get }
    var outputAmount:Int { get }

    var aValues:[Tensor<Float>] { get set }
    var zValues:[Tensor<Float>] { get set }
    
    var derivativeAmount: Int { get }

    mutating func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>]
    func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>])
    
}


