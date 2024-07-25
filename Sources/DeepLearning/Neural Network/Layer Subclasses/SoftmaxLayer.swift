//
//  SoftmaxLayer.swift
//  DeepLearning
//
//  Created by Max Gasslitter Strobl on 16/07/2020.
//  Copyright © 2020 Max Gasslitter Strobl. All rights reserved.
//

import Foundation

final internal class SoftmaxLayer: Layer {
    
    var logValues: [Tensor<Float>]
    
    override func propagateForeward(inputs: [Tensor<Float>]) -> [Tensor<Float>] {
        
        assert(inputs.count == self.inputAmount, "Incorrect Input Shape")
        
        self.zValues = inputs
        self.aValues = inputs
        let results = zValues.map { return self.softmax($0) }
        
        self.newZValues = results.map { return $0.sm }
        //self.logValues = results.map { return $0.logsm }
        
        return self.newZValues
        
    }
    
    private func softmax(_ inputs: Tensor<Float>) -> (sm: Tensor<Float>, logsm: Tensor<Float>?) {
        
        //var output = Tensor<Float>(inputs.shape, constant: 0)
        
        let normalizer:Float = inputs.findMax()
        let exponentiated = (inputs - normalizer).exp()
        let sum = exponentiated.sum()
        let sm = (exponentiated / sum).fast()
        
        /*let logSum = -log(sum)
        let logsm = inputs + logSum*/
        
        return (sm, nil)
        
        /*DispatchQueue.concurrentPerform(iterations: inputs.elements.count) { (i) in

            let j = inputs.elements[i]
            
            if j > max {
                
                max = j
                
            }
            
        }*/
        
        //var total:Float = 0.0
        
        /*DispatchQueue.concurrentPerform(iterations: inputs.elements.count) { (i) in

            let j = inputs.elements[i]
            
            total += exp(j - max)
            
        }
        
        for i in inputs.elements.indices {
            
            output.elements[i] = exp(inputs.elements[i] - max)/total
            
        }
        
        return output*/
        
    }
    
    override func propagateBackward(derivatives: [Tensor<Float>]) -> (newDerivatives: [Tensor<Float>], internalDerivatives: [Tensor<Float>]) {
        
        assert(derivatives.count == self.outputAmount, "Derivatives are of the Incorrect Shape")
        
        return (newDerivatives: derivatives.enumerated().map { (i, derivative) in
            let der = self.derivativeSoftmax(self.newZValues[i], self.aValues[i])
            let output = (der * derivative).fast()
            return output },
                internalDerivatives: [])
        
    }
    
    private func derivativeSoftmax(_ aValues: Tensor<Float>,_ zValues: Tensor<Float>) -> Tensor<Float> {
        
        let shape = Shape([self.inputShape[0],self.outputShape[0]])
        
        /*var output = Tensor<Float>(shape, constant: 0)
        
        let testTimeStart1 = CFAbsoluteTimeGetCurrent()
        
        let inputCount = zValues.elements.count
        
        output.elements = output.elements.enumerated().map({ (i, output) in
            
            let j = Int(i) % inputCount
            let i = Int(floor(Double(i / inputCount)))
            
            let aValue = aValues.elements[i]
            
            let iIsJ:Float = i == j ? 1 : 0
            let result = aValue * (iIsJ - aValues.elements[j])
            
            return result
            
        })
        
        print("diff 1: \(CFAbsoluteTimeGetCurrent() - testTimeStart1)")*/
        
        /*var outputTest = Tensor<Float>(shape, constant: 0)
        
         for i in 0...shape[0].value - 1 {
             
             for j in 0...shape[1].value - 1 {
                 
                 let aValueI = aValues[i]
                 
                 if i == j {
                     
                     outputTest[i, j] = aValueI * (1 - aValueI)
                     
                 }else{
                     
                     let aValueJ = aValues[j]
                     outputTest[i, j] = -aValueI * aValueJ
                     
                 }
                 
             }
             
         }*/
        
        var outputTest1 = Tensor<Float>(shape, constant: 0)
        
        //let testTimeStart2 = CFAbsoluteTimeGetCurrent()
        
        outputTest1 = (-1 * aValues * aValues).fast()
        
        let _ = aValues.elements.enumerated().map { (i, value) in
            
            outputTest1[i, i] = value * (1 - value)
            return 0
            
        }
        
        //print("diff 2: \(CFAbsoluteTimeGetCurrent() - testTimeStart2)")
        
        //print(output == outputTest)
        //print(output == outputTest1)
        //print(outputTest == outputTest1)
        
        return outputTest1
        
    }
    
    init(features: Int) {
        
        self.logValues = []
        
        let shape = Shape([Dimension(features)])
        
        super.init(inputShape: shape, outputShape: shape, activationFunction: .normal)
        
    }
    
    required init(from decoder: Decoder) throws {
        
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let features = try values.decode(Int.self, forKey: .features)
        
        self.logValues = []
        
        let shape = Shape([Dimension(features)])
        super.init(inputShape: shape, outputShape: shape, activationFunction: .normal)

        
    }
    
    override func encode(to encoder: Encoder) throws {
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.inputShape[0].value, forKey: .features)
        
        try container.encode(Network.InternalLayerType.softmax, forKey: .type)

    }
    
    override func copy() -> SoftmaxLayer {
        
        let newLayer = SoftmaxLayer(features: self.inputShape.volume)
        
        return newLayer
        
    }
    
    private enum CodingKeys: String, CodingKey {
        
        case features
        case type

    }
    
}
