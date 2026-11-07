package com.example.t2imobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.os.SystemClock
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random

enum class Provider { CPU, XNNPACK, NNAPI }

data class Timings(
    var loadMs: Long = 0, var tokenizeMs: Long = 0, var textMs: Long = 0,
    var unetMs: Long = 0, var vaeMs: Long = 0, var totalMs: Long = 0,
    var peakPssMb: Long = 0, var peakNativeMb: Long = 0, var providerNote: String = ""
)

data class GenResult(val bitmap: Bitmap, val timings: Timings, val steps: Int, val size: Int)

/**
 * Text encoder -> LCM UNet loop -> VAE decoder on ONNX Runtime.
 *
 * lowMemory=true loads each model just-in-time and closes it afterwards
 * (lower peak RAM, extra load time).
 *
 * When guidance is enabled, classifier-free guidance runs the UNet separately for the negative
 * and positive embeddings, then combines their noise predictions. Separate calls keep the
 * pipeline compatible with the existing fixed-batch ONNX models.
 *
 * Model dir layout:
 *   root/{vocab.json, merges.txt, config.json, <variant>/{text_encoder, unet, vae_decoder}.onnx}
 */
class T2IPipeline(
    private val root: File,
    val variant: String,
    val provider: Provider,
    val lowMemory: Boolean
) : AutoCloseable {

    private val env  = OrtEnvironment.getEnvironment()
    private val dir  = File(root, variant)
    private val cfg  = JSONObject(File(root, "config.json").readText())
    private val vaeScale = cfg.getDouble("vae_scale").toFloat()
    private val hidden   = cfg.getInt("hidden")
    private val tokenizer = ClipTokenizer(
        File(root, "vocab.json").readText(),
        File(root, "merges.txt").readText(),
        cfg.getInt("bos"),
        cfg.getInt("eos")
    )
    private var text: OrtSession? = null
    private var unet: OrtSession? = null
    private var vae:  OrtSession? = null
    private var note = ""

    private fun options(): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(4)
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        try {
            when (provider) {
                Provider.XNNPACK -> o.addXnnpack(mapOf("intra_op_num_threads" to "4"))
                Provider.NNAPI   -> o.addNnapi()
                Provider.CPU     -> {}
            }
        } catch (e: Exception) {
            note = "${provider.name} unavailable (${e.message}); using CPU"
        }
        return o
    }

    private fun load(name: String): OrtSession =
        env.createSession(File(dir, "$name.onnx").absolutePath, options())

    private fun FloatBuffer.toArray(): FloatArray = FloatArray(remaining()).also { get(it) }

    /**
     * Generate one image.
     *
     * @param prompt         The positive text prompt.
     * @param steps          Number of LCM denoising steps (1–8).
     * @param size           Output resolution in pixels (256 / 384 / 512).
     * @param seed           Random seed for the initial latent noise.
     * @param negativePrompt Optional negative prompt used when guidanceScale > 1.
     * @param guidanceScale  LCM classifier-free guidance strength; 0 disables guidance.
     * @param onStage        Progress callback — called with a human-readable stage name.
     */
    fun generate(
        prompt: String,
        steps: Int,
        size: Int,
        seed: Long,
        negativePrompt: String? = null,
        guidanceScale: Float = 0f,
        onStage: (String) -> Unit = {}
    ): GenResult {
        val t   = Timings()
        val mon = MemoryMonitor().also { it.start() }
        val t0  = SystemClock.elapsedRealtime()
        try {
            // ── tokenise ──────────────────────────────────────────────────
            var s   = SystemClock.elapsedRealtime()
            val ids = tokenizer.encode(prompt)
            val useGuidance = guidanceScale > 1f
            val negIds = if (useGuidance) {
                tokenizer.encode(negativePrompt.orEmpty())
            } else {
                null
            }
            t.tokenizeMs = SystemClock.elapsedRealtime() - s

            // ── text encoder ──────────────────────────────────────────────
            onStage("text encoder")
            s   = SystemClock.elapsedRealtime()
            val ts  = text ?: load("text_encoder").also { text = it }
            t.loadMs += SystemClock.elapsedRealtime() - s
            s   = SystemClock.elapsedRealtime()

            // Positive embedding (shape 1×77×768)
            val idT  = OnnxTensor.createTensor(env,
                LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()))
            val posEmb: FloatArray = ts.run(mapOf("input_ids" to idT))
                .use { (it[0] as OnnxTensor).floatBuffer.toArray() }
            idT.close()

            val negEmb = negIds?.let { nids ->
                val nidT = OnnxTensor.createTensor(env,
                    LongBuffer.wrap(nids), longArrayOf(1, nids.size.toLong()))
                ts.run(mapOf("input_ids" to nidT))
                    .use { (it[0] as OnnxTensor).floatBuffer.toArray() }
                    .also { nidT.close() }
            }

            t.textMs = SystemClock.elapsedRealtime() - s
            if (lowMemory) { ts.close(); text = null }

            // ── UNet (LCM loop) ──────────────────────────────────────────
            s      = SystemClock.elapsedRealtime()
            val us = unet ?: load("unet").also { unet = it }
            t.loadMs += SystemClock.elapsedRealtime() - s

            val lat   = size / 8
            val shape = longArrayOf(1, 4, lat.toLong(), lat.toLong())
            val rnd   = Random(seed)
            var x     = FloatArray(4 * lat * lat) { rnd.nextGaussian().toFloat() }
            val sched = LcmScheduler(steps)
            val posEmbT  = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(posEmb),
                longArrayOf(1, ids.size.toLong(), hidden.toLong()))
            val negEmbT = negEmb?.let {
                OnnxTensor.createTensor(env, FloatBuffer.wrap(it),
                    longArrayOf(1, ids.size.toLong(), hidden.toLong()))
            }

            s = SystemClock.elapsedRealtime()
            try {
                for (i in 0 until steps) {
                    onStage("UNet step ${i + 1}/$steps")
                    val timestep = sched.timesteps[i].toFloat()
                    val conditional = predict(us, x, shape, timestep, posEmbT)
                    val eps = if (negEmbT != null) {
                        val unconditional = predict(us, x, shape, timestep, negEmbT)
                        FloatArray(conditional.size) { j ->
                            unconditional[j] + guidanceScale * (conditional[j] - unconditional[j])
                        }
                    } else {
                        conditional
                    }
                    x = sched.step(eps, i, x, rnd)
                }
            } finally {
                posEmbT.close()
                negEmbT?.close()
            }
            t.unetMs = SystemClock.elapsedRealtime() - s
            if (lowMemory) { us.close(); unet = null }

            // ── VAE decoder ───────────────────────────────────────────────
            onStage("VAE decode")
            s     = SystemClock.elapsedRealtime()
            val vs = vae ?: load("vae_decoder").also { vae = it }
            t.loadMs += SystemClock.elapsedRealtime() - s
            s     = SystemClock.elapsedRealtime()
            val z = FloatArray(x.size) { x[it] / vaeScale }
            val zT = OnnxTensor.createTensor(env, FloatBuffer.wrap(z), shape)
            val img = vs.run(mapOf("latent_sample" to zT))
                .use { (it[0] as OnnxTensor).floatBuffer.toArray() }
            zT.close()
            t.vaeMs = SystemClock.elapsedRealtime() - s
            if (lowMemory) { vs.close(); vae = null }

            val bmp = toBitmap(img, size)
            t.totalMs = SystemClock.elapsedRealtime() - t0
            return GenResult(bmp, t, steps, size)

        } finally {
            mon.stop()
            t.peakPssMb    = mon.peakPssMb
            t.peakNativeMb = mon.peakNativeMb
            t.providerNote = note
        }
    }

    private fun predict(
        session: OrtSession,
        sample: FloatArray,
        sampleShape: LongArray,
        timestep: Float,
        embedding: OnnxTensor
    ): FloatArray {
        val sampleTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), sampleShape)
        val timestepTensor = OnnxTensor.createTensor(env,
            FloatBuffer.wrap(floatArrayOf(timestep)), longArrayOf(1))
        return try {
            session.run(mapOf(
                "sample" to sampleTensor,
                "timestep" to timestepTensor,
                "encoder_hidden_states" to embedding
            )).use { (it[0] as OnnxTensor).floatBuffer.toArray() }
        } finally {
            sampleTensor.close()
            timestepTensor.close()
        }
    }

    /** NCHW float [0,1] -> ARGB Bitmap */
    private fun toBitmap(chw: FloatArray, size: Int): Bitmap {
        val n  = size * size
        val px = IntArray(n)
        fun c(v: Float) = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        for (i in 0 until n)
            px[i] = (0xFF shl 24) or (c(chw[i]) shl 16) or
                    (c(chw[n + i]) shl 8) or c(chw[2 * n + i])
        return Bitmap.createBitmap(px, size, size, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        text?.close(); unet?.close(); vae?.close()
        text = null; unet = null; vae = null
    }
}
