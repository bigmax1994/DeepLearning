//
//  Dimension.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

public struct Dimension: Codable, Equatable {
    
    public let value: Int
    
    public init(_ value: Int) {
        
        assert(value > 0, "Dimension must be greater than 0")
        
        self.value = value
        
    }
    
}

extension Dimension: ExpressibleByIntegerLiteral {
    
    public init(integerLiteral value: Int) {
        
        self.init(value)
        
    }
    
}

extension Int {
    internal func ceilDiv(_ rhs: Int) -> Int {
        return (self + rhs - 1) / rhs
    }
}
