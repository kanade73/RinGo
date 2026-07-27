private extension ConvLayerDesc {
    mutating func scaleOutputChannels(_ scaling: [Float]) {
        precondition(scaling.count == outChannels)
        let outputStride = convYSize * convXSize * inChannels
        for output in 0 ..< outChannels {
            let factor = scaling[output]
            let start = output * outputStride
            for index in start ..< start + outputStride {
                weights[index] *= factor
            }
        }
    }

    mutating func unscaleOutputChannels(_ scaling: [Float]) {
        precondition(scaling.count == outChannels)
        let outputStride = convYSize * convXSize * inChannels
        for output in 0 ..< outChannels {
            let factor = scaling[output]
            let start = output * outputStride
            for index in start ..< start + outputStride {
                weights[index] /= factor
            }
        }
    }
}

private extension MatMulLayerDesc {
    mutating func scaleOutputChannels(_ scaling: [Float]) {
        precondition(scaling.count == outChannels)
        for input in 0 ..< inChannels {
            for output in 0 ..< outChannels {
                weights[input * outChannels + output] *= scaling[output]
            }
        }
    }

    mutating func unscaleOutputChannels(_ scaling: [Float]) {
        precondition(scaling.count == outChannels)
        for input in 0 ..< inChannels {
            for output in 0 ..< outChannels {
                weights[input * outChannels + output] /= scaling[output]
            }
        }
    }
}

private extension BatchNormLayerDesc {
    mutating func scaleInputChannels(_ scaling: [Float]) {
        precondition(scaling.count == numChannels)
        epsilon = 1e-20
        for channel in 0 ..< numChannels {
            mergedScale[channel] *= scaling[channel]
            mean[channel] = 0
            variance[channel] = 1 - epsilon
            scale[channel] = mergedScale[channel]
            bias[channel] = mergedBias[channel]
        }
    }

    mutating func extractChannelFactorsAbsLessThanOne() -> [Float] {
        epsilon = 1e-20
        var channelFactors = [Float](repeating: 1, count: numChannels)
        for channel in 0 ..< numChannels {
            if abs(mergedScale[channel]) < 1 {
                channelFactors[channel] = mergedScale[channel]
                mergedScale[channel] = 1
            }
            mean[channel] = 0
            variance[channel] = 1 - epsilon
            scale[channel] = mergedScale[channel]
            bias[channel] = mergedBias[channel]
        }
        return channelFactors
    }

    mutating func extractChannelFactorsWithInverses() -> (factors: [Float], inverses: [Float]) {
        epsilon = 1e-20
        var channelFactors = [Float](repeating: 1, count: numChannels)
        var inverseFactors = channelFactors
        for channel in 0 ..< numChannels {
            if abs(mergedScale[channel]) < 0.5 {
                channelFactors[channel] = 0.5
                inverseFactors[channel] = 2
                mergedScale[channel] *= 2
            } else if abs(mergedScale[channel]) < 1 {
                channelFactors[channel] = mergedScale[channel]
                inverseFactors[channel] = 1 / mergedScale[channel]
                mergedScale[channel] = 1
            }
            mean[channel] = 0
            variance[channel] = 1 - epsilon
            scale[channel] = mergedScale[channel]
            bias[channel] = mergedBias[channel]
        }
        return (channelFactors, inverseFactors)
    }

    mutating func inverseExtractChannelFactorsAbsLessThanOne() throws {
        try requireTransformCanonicalForm()
        guard mergedScale.allSatisfy({ abs($0) >= 1 }) else {
            throw ModelWriteError(
                "\(name): descriptor is not inference-ready; activation-reduced BatchNorm scale has magnitude below 1"
            )
        }
        try setDiskMergedValues(scale: mergedScale, bias: mergedBias)
    }

    mutating func inverseExtractChannelFactorsWithInverses() throws -> (
        factors: [Float],
        inverses: [Float]
    ) {
        try requireTransformCanonicalForm()
        var diskScale = mergedScale
        var factors = [Float](repeating: 1, count: numChannels)
        var inverses = factors
        for channel in 0 ..< numChannels where abs(mergedScale[channel]) < 1 {
            diskScale[channel] *= 0.5
            factors[channel] = 0.5
            inverses[channel] = 2
        }
        try setDiskMergedValues(scale: diskScale, bias: mergedBias)
        return (factors, inverses)
    }

    mutating func inverseScaleInputChannels(_ scaling: [Float]) throws {
        try requireTransformCanonicalForm()
        let diskScale = zip(mergedScale, scaling).map(/)
        try setDiskMergedValues(scale: diskScale, bias: mergedBias)
    }

    func requireTransformCanonicalForm() throws {
        guard mean.count == numChannels, variance.count == numChannels,
              scale.count == numChannels, bias.count == numChannels,
              mergedScale.count == numChannels, mergedBias.count == numChannels
        else {
            throw ModelWriteError("\(name): inconsistent BatchNorm channel-array lengths")
        }
        guard epsilon == 1e-20, mean.allSatisfy({ $0 == 0 }), variance.allSatisfy({ $0 == 1 }),
              scale == mergedScale, bias == mergedBias
        else {
            throw ModelWriteError(
                "\(name): descriptor is not in the inference-ready BatchNorm form produced by the load transform"
            )
        }
    }

    mutating func setDiskMergedValues(scale desiredScale: [Float], bias desiredBias: [Float]) throws {
        epsilon = 1e-20
        mean = [Float](repeating: 0, count: numChannels)
        variance = [Float](repeating: 1, count: numChannels)
        scale = desiredScale
        bias = desiredBias
        mergedScale = desiredScale
        mergedBias = desiredBias

        for channel in 0 ..< numChannels {
            if !hasScale {
                let desired = desiredScale[channel]
                guard desired > 0, desired.isFinite else {
                    throw ModelWriteError(
                        "\(name): cannot preserve hasScale=false for nonpositive merged scale \(desired)"
                    )
                }
                variance[channel] = 1 / (desired * desired) - epsilon
                scale[channel] = 1
            }
            if !hasBias {
                mean[channel] = -desiredBias[channel] / desiredScale[channel]
                bias[channel] = 0
            }
        }
    }
}

private extension ResidualBlockDesc {
    mutating func inverseTransformToReduceActivations() throws {
        try midBN.inverseExtractChannelFactorsAbsLessThanOne()
    }
}

private extension GlobalPoolingResidualBlockDesc {
    mutating func inverseTransformToReduceActivations() throws {
        try gpoolBN.inverseExtractChannelFactorsAbsLessThanOne()
        try midBN.inverseExtractChannelFactorsAbsLessThanOne()
    }
}

private extension NestedBottleneckResidualBlockDesc {
    mutating func inverseTransformToReduceActivations() throws {
        try blocks.inverseTransformChildrenToReduceActivations()
        let hasImmediateTransformer = blocks.contains {
            $0.kind == .transformerAttention || $0.kind == .transformerFFN
        }
        if !hasImmediateTransformer {
            let extracted = try postBN.inverseExtractChannelFactorsWithInverses()
            preConv.unscaleOutputChannels(extracted.factors)
            for index in blocks.indices {
                switch blocks[index] {
                case var .ordinary(block):
                    try block.preBN.inverseScaleInputChannels(extracted.inverses)
                    block.finalConv.unscaleOutputChannels(extracted.factors)
                    blocks[index] = .ordinary(block)
                case var .globalPooling(block):
                    try block.preBN.inverseScaleInputChannels(extracted.inverses)
                    block.finalConv.unscaleOutputChannels(extracted.factors)
                    blocks[index] = .globalPooling(block)
                case var .nestedBottleneck(block):
                    try block.preBN.inverseScaleInputChannels(extracted.inverses)
                    block.postConv.unscaleOutputChannels(extracted.factors)
                    blocks[index] = .nestedBottleneck(block)
                case .transformerAttention, .transformerFFN:
                    preconditionFailure("Transformer child was excluded from nested inverse scaling")
                }
            }
        }
    }
}

private extension [BlockDesc] {
    mutating func inverseTransformChildrenToReduceActivations() throws {
        for index in indices {
            switch self[index] {
            case var .ordinary(block):
                try block.inverseTransformToReduceActivations()
                self[index] = .ordinary(block)
            case var .globalPooling(block):
                try block.inverseTransformToReduceActivations()
                self[index] = .globalPooling(block)
            case var .nestedBottleneck(block):
                try block.inverseTransformToReduceActivations()
                self[index] = .nestedBottleneck(block)
            case .transformerAttention, .transformerFFN:
                break
            }
        }
    }
}

private extension TrunkDesc {
    mutating func inverseTransformToReduceActivations() throws {
        try blocks.inverseTransformChildrenToReduceActivations()
        let hasImmediateTransformer = blocks.contains {
            $0.kind == .transformerAttention || $0.kind == .transformerFFN
        }
        if trunkNormKind == .standard, !hasImmediateTransformer {
            guard var tipNorm = trunkTipBN else {
                throw ModelWriteError("\(name): standard trunk is missing its tip BatchNorm")
            }
            let extracted = try tipNorm.inverseExtractChannelFactorsWithInverses()
            trunkTipBN = tipNorm
            initialConv.unscaleOutputChannels(extracted.factors)
            initialMatMul.unscaleOutputChannels(extracted.factors)
            for index in blocks.indices {
                switch blocks[index] {
                case var .ordinary(block):
                    try block.preBN.inverseScaleInputChannels(extracted.inverses)
                    block.finalConv.unscaleOutputChannels(extracted.factors)
                    blocks[index] = .ordinary(block)
                case var .globalPooling(block):
                    try block.preBN.inverseScaleInputChannels(extracted.inverses)
                    block.finalConv.unscaleOutputChannels(extracted.factors)
                    blocks[index] = .globalPooling(block)
                case var .nestedBottleneck(block):
                    try block.preBN.inverseScaleInputChannels(extracted.inverses)
                    block.postConv.unscaleOutputChannels(extracted.factors)
                    blocks[index] = .nestedBottleneck(block)
                case .transformerAttention, .transformerFFN:
                    preconditionFailure("Transformer child was excluded from trunk inverse scaling")
                }
            }
        }
    }
}

private extension PolicyHeadDesc {
    mutating func inverseTransformToReduceActivations() throws {
        try g1BN.inverseExtractChannelFactorsAbsLessThanOne()
        try p1BN.inverseExtractChannelFactorsAbsLessThanOne()
    }
}

private extension ValueHeadDesc {
    mutating func inverseTransformToReduceActivations() throws {
        try v1BN.inverseExtractChannelFactorsAbsLessThanOne()
    }
}

private extension ResidualBlockDesc {
    mutating func transformToReduceActivations() {
        let factors = midBN.extractChannelFactorsAbsLessThanOne()
        regularConv.scaleOutputChannels(factors)
    }
}

private extension GlobalPoolingResidualBlockDesc {
    mutating func transformToReduceActivations() {
        var factors = midBN.extractChannelFactorsAbsLessThanOne()
        regularConv.scaleOutputChannels(factors)
        gpoolToBiasMul.scaleOutputChannels(factors)
        factors = gpoolBN.extractChannelFactorsAbsLessThanOne()
        gpoolConv.scaleOutputChannels(factors)
    }
}

private extension NestedBottleneckResidualBlockDesc {
    mutating func transformToReduceActivations() {
        let hasImmediateTransformer = blocks.contains {
            $0.kind == .transformerAttention || $0.kind == .transformerFFN
        }
        if !hasImmediateTransformer {
            let extracted = postBN.extractChannelFactorsWithInverses()
            preConv.scaleOutputChannels(extracted.factors)
            for index in blocks.indices {
                switch blocks[index] {
                case var .ordinary(block):
                    block.preBN.scaleInputChannels(extracted.inverses)
                    block.finalConv.scaleOutputChannels(extracted.factors)
                    blocks[index] = .ordinary(block)
                case var .globalPooling(block):
                    block.preBN.scaleInputChannels(extracted.inverses)
                    block.finalConv.scaleOutputChannels(extracted.factors)
                    blocks[index] = .globalPooling(block)
                case var .nestedBottleneck(block):
                    block.preBN.scaleInputChannels(extracted.inverses)
                    block.postConv.scaleOutputChannels(extracted.factors)
                    blocks[index] = .nestedBottleneck(block)
                case .transformerAttention, .transformerFFN:
                    preconditionFailure("Transformer child was excluded from nested scaling")
                }
            }
        }
        blocks.transformChildrenToReduceActivations()
    }
}

private extension [BlockDesc] {
    mutating func transformChildrenToReduceActivations() {
        for index in indices {
            switch self[index] {
            case var .ordinary(block):
                block.transformToReduceActivations()
                self[index] = .ordinary(block)
            case var .globalPooling(block):
                block.transformToReduceActivations()
                self[index] = .globalPooling(block)
            case var .nestedBottleneck(block):
                block.transformToReduceActivations()
                self[index] = .nestedBottleneck(block)
            case .transformerAttention, .transformerFFN:
                break
            }
        }
    }
}

private extension TrunkDesc {
    mutating func transformToReduceActivations() {
        let hasImmediateTransformer = blocks.contains {
            $0.kind == .transformerAttention || $0.kind == .transformerFFN
        }
        if trunkNormKind == .standard, !hasImmediateTransformer {
            guard var tipNorm = trunkTipBN else {
                preconditionFailure("Standard trunk is missing its tip BatchNorm")
            }
            let extracted = tipNorm.extractChannelFactorsWithInverses()
            trunkTipBN = tipNorm
            initialConv.scaleOutputChannels(extracted.factors)
            initialMatMul.scaleOutputChannels(extracted.factors)
            if var metadataEncoder = sgfMetadataEncoder {
                metadataEncoder.mul3.scaleOutputChannels(extracted.factors)
                sgfMetadataEncoder = metadataEncoder
            }
            for index in blocks.indices {
                switch blocks[index] {
                case var .ordinary(block):
                    block.preBN.scaleInputChannels(extracted.inverses)
                    block.finalConv.scaleOutputChannels(extracted.factors)
                    blocks[index] = .ordinary(block)
                case var .globalPooling(block):
                    block.preBN.scaleInputChannels(extracted.inverses)
                    block.finalConv.scaleOutputChannels(extracted.factors)
                    blocks[index] = .globalPooling(block)
                case var .nestedBottleneck(block):
                    block.preBN.scaleInputChannels(extracted.inverses)
                    block.postConv.scaleOutputChannels(extracted.factors)
                    blocks[index] = .nestedBottleneck(block)
                case .transformerAttention, .transformerFFN:
                    preconditionFailure("Transformer child was excluded from trunk scaling")
                }
            }
        }
        blocks.transformChildrenToReduceActivations()
    }
}

private extension PolicyHeadDesc {
    mutating func transformToReduceActivations() {
        var factors = p1BN.extractChannelFactorsAbsLessThanOne()
        p1Conv.scaleOutputChannels(factors)
        gpoolToBiasMul.scaleOutputChannels(factors)
        factors = g1BN.extractChannelFactorsAbsLessThanOne()
        g1Conv.scaleOutputChannels(factors)
    }
}

private extension ValueHeadDesc {
    mutating func transformToReduceActivations() {
        let factors = v1BN.extractChannelFactorsAbsLessThanOne()
        v1Conv.scaleOutputChannels(factors)
    }
}

extension ModelDesc {
    mutating func transformToReduceActivations() {
        trunk.transformToReduceActivations()
        policyHead.transformToReduceActivations()
        valueHead.transformToReduceActivations()
    }

    mutating func inverseTransformToReduceActivationsForWriting() throws {
        try trunk.inverseTransformToReduceActivations()
        try policyHead.inverseTransformToReduceActivations()
        try valueHead.inverseTransformToReduceActivations()
    }
}
