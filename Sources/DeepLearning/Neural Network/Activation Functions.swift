//
//  Activation Functions.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

public enum ActivationFunction: String, Codable, Equatable {
    
    case normal
    case positive
    case relu
    case smoothRelu
    case leakyRelu
    case sigmoid
    case arctan
    case tanh
    
}

public enum LossFunction: String, Codable, Equatable {
    
    case meanErrorSquared
    case crossEntropy
    
}
