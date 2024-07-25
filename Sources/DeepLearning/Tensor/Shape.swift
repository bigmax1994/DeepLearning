//
//  Shape.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 14/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

public struct Shape: Equatable, Codable {
    
    public let dimensions: [Dimension]
    
    public func transposed() -> Shape {
        
        if self.dimensions.count != 2 {
            return Shape([0])
        }
            
        return Shape(arrayLiteral: self[1],self[0])
        
    }
    
    public var volume: Int {
        
        get {
            
            var volume = dimensions[0].value
            
            if self.dimensions.count == 1 { return volume }
            
            for i in 1..<self.dimensions.count { volume *= self[i].value }
            
            return volume
            
        }
        
    }
    
    public init(_ dimensions: [Dimension]) {
        
        self.dimensions = dimensions
        
    }
    
    public init(_ values: [Int]) {
        
        let dimensions = values.map { return Dimension($0) }
        
        self.init(dimensions)
        
    }
    
    public subscript (_ index: Int) -> Dimension {
        
        get {
            
            assert(index >= 0 && index < self.dimensions.count, "Index out of Range")
            
            return self.dimensions[index]
            
        }
        
    }
    
}

extension Shape: ExpressibleByArrayLiteral {
    
    public init(arrayLiteral elements: Dimension...) {
        
        self.init(elements)
        
    }
    
}
