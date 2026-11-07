package com.example.t2imobile

import java.util.Random
import kotlin.math.sqrt

/** Port of pc/common.py LCMScheduler (epsilon prediction). 4 steps -> 999, 759, 519, 279. */
class LcmScheduler(steps: Int, originalSteps: Int = 50,
                   private val timestepScaling: Double = 10.0,
                   private val sigmaData: Double = 0.5) {

    private val alphasCumprod = DoubleArray(1000)
    val timesteps: IntArray

    init {
        val b0 = sqrt(0.00085); val b1 = sqrt(0.012)
        var prod = 1.0
        for (i in 0 until 1000) {
            val b = b0 + (b1 - b0) * i / 999.0
            prod *= 1.0 - b * b
            alphasCumprod[i] = prod
        }
        val k = 1000 / originalSteps
        val origin = IntArray(originalSteps) { (it + 1) * k - 1 }
        val skip = origin.size / steps
        timesteps = IntArray(steps) { origin[origin.size - 1 - it * skip] }
    }

    /** One LCM step. [eps] = UNet output, [sample] = current latents. Returns the next latents. */
    fun step(eps: FloatArray, i: Int, sample: FloatArray, rnd: Random): FloatArray {
        val t = timesteps[i]
        val last = i + 1 >= timesteps.size
        val aT = alphasCumprod[t]
        val aPrev = if (last) 1.0 else alphasCumprod[timesteps[i + 1]]
        val st = t * timestepScaling
        val cSkip = sigmaData * sigmaData / (st * st + sigmaData * sigmaData)
        val cOut = st / sqrt(st * st + sigmaData * sigmaData)
        val sqA = sqrt(aT); val sq1A = sqrt(1 - aT)
        val sqPrev = sqrt(aPrev); val sq1Prev = sqrt(1 - aPrev)
        val out = FloatArray(sample.size)
        for (j in sample.indices) {
            val x0 = (sample[j] - sq1A * eps[j]) / sqA
            val denoised = cOut * x0 + cSkip * sample[j]
            out[j] = if (last) denoised.toFloat()
            else (sqPrev * denoised + sq1Prev * rnd.nextGaussian()).toFloat()
        }
        return out
    }
}
