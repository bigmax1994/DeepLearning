//
//  File.swift
//  
//
//  Created by Max Gasslitter Strobl on 03/09/22.
//

import Foundation

class GaussianDistribution {
    let mean: Float
    let deviation: Float

    init(mean: Float, deviation: Float) {
        precondition(deviation >= 0)
        self.mean = mean
        self.deviation = deviation
    }

    func nextFloat() -> Float {
        guard deviation > 0 else { return mean }

        let x1 = Float.random(in: 0...1) // a random number between 0 and 1
        let x2 = Float.random(in: 0...1) // a random number between 0 and 1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2) // z1 is normally distributed

        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * deviation + mean
    }
}
