//
//  Algebra.swift
//
//  Created by strictlyswift on 9/8/17.
//

import Foundation


public protocol Semiring {
    static func + (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static var zero: Self { get }
    static var one: Self { get }
}

public protocol Ring : Semiring {
    static func - (lhs: Self, rhs: Self) -> Self
    
}

extension Semiring {
    static func += (lhs: inout Self, rhs: Self) {
        lhs = lhs + rhs
    }
}

extension Float: Ring {
    public static var zero: Float {
        return 0
    }
    
    public static var one: Float {
        return 1
    }
}

extension Int : Ring {
    public static var zero: Int {
        return 0
    }
    
    public static var one: Int {
        return 1
    }
}

extension Bool : Semiring {
    public static func + (lhs: Bool, rhs: Bool) -> Bool {
        return lhs || rhs
    }
    
    public static func * (lhs: Bool, rhs: Bool) -> Bool {
        return lhs && rhs
    }
    
    public static let zero = false
    public static let one = true
    
}
