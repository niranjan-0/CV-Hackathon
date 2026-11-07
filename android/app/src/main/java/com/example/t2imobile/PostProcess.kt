package com.example.t2imobile

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/** Classical CV: image quality assessment, artifact detection/correction, edge-aware enhancement, resizing. */
object PostProcess {

    data class Quality(
        val sharpness: Double, val blackFrac: Double, val whiteFrac: Double,
        val contrast: Double, val castA: Double, val castB: Double, val flags: List<String>
    ) {
        override fun toString() =
            "sharp=%.0f contrast=%.0f black=%.2f white=%.2f cast=(%.1f,%.1f) flags=%s".format(
                sharpness, contrast, blackFrac, whiteFrac, castA, castB, if (flags.isEmpty()) "none" else flags.joinToString())
    }

    private fun toRgb(bmp: Bitmap): Mat {
        val rgba = Mat(); Utils.bitmapToMat(bmp, rgba)
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        return rgb
    }

    private fun toBitmap(rgb: Mat): Bitmap {
        val rgba = Mat(); Imgproc.cvtColor(rgb, rgba, Imgproc.COLOR_RGB2RGBA)
        val bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }

    private fun assess(rgb: Mat): Quality {
        val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
        val lap = Mat(); Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mu = MatOfDouble(); val sd = MatOfDouble()
        Core.meanStdDev(lap, mu, sd)
        val sharp = sd.toArray()[0].let { it * it }                      // variance of Laplacian
        Core.meanStdDev(gray, mu, sd)
        val contrast = sd.toArray()[0]
        val total = gray.total().toDouble()
        val m = Mat()
        Imgproc.threshold(gray, m, 10.0, 255.0, Imgproc.THRESH_BINARY_INV)
        val black = Core.countNonZero(m) / total
        Imgproc.threshold(gray, m, 245.0, 255.0, Imgproc.THRESH_BINARY)
        val white = Core.countNonZero(m) / total
        val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val mean = Core.mean(lab)
        val ca = mean.`val`[1] - 128.0; val cb = mean.`val`[2] - 128.0
        val flags = ArrayList<String>()
        if (sharp < 60) flags += "blurry"
        if (black > 0.4) flags += "dark_region"
        if (white > 0.3) flags += "blown_highlights"
        if (contrast < 25) flags += "low_contrast"
        if (maxOf(abs(ca), abs(cb)) > 8) flags += "color_cast"
        gray.release(); lap.release(); m.release(); lab.release()
        return Quality(sharp, black, white, contrast, ca, cb, flags)
    }

    fun assess(bmp: Bitmap): Quality = toRgb(bmp).let { val q = assess(it); it.release(); q }

    /**
     * Pipeline: gray-world WB (if cast) -> CLAHE on L -> highlight gamma (if blown) -> bilateral denoise
     * -> edge-aware unsharp mask (Canny edge mask) -> optional Lanczos resize to [target].
     * Returns the processed bitmap with before/after quality.
     */
    fun enhance(bmp: Bitmap, target: Int? = null): Triple<Bitmap, Quality, Quality> {
        var rgb = toRgb(bmp)
        val before = assess(rgb)
        val flags = before.flags

        if ("color_cast" in flags) {                                     // gray-world white balance
            val ch = ArrayList<Mat>(); Core.split(rgb, ch)
            val means = ch.map { Core.mean(it).`val`[0] }
            val g = means.average()
            for (i in 0 until 3) Core.multiply(ch[i], Scalar(g / maxOf(means[i], 1e-3)), ch[i])
            val merged = Mat(); Core.merge(ch, merged); rgb.release(); rgb = merged
            ch.forEach { it.release() }
        }

        run {                                                            // CLAHE on lightness
            val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
            val ch = ArrayList<Mat>(); Core.split(lab, ch)
            val clahe: CLAHE = Imgproc.createCLAHE(if ("low_contrast" in flags) 2.5 else 1.5, Size(8.0, 8.0))
            clahe.apply(ch[0], ch[0])
            Core.merge(ch, lab)
            Imgproc.cvtColor(lab, rgb, Imgproc.COLOR_Lab2RGB)
            lab.release(); ch.forEach { it.release() }
        }

        if ("blown_highlights" in flags) {                               // artifact correction: tame highlights
            val lut = Mat(1, 256, CvType.CV_8U)
            val data = ByteArray(256) { (255.0 * Math.pow(it / 255.0, 1.2)).toInt().toByte() }
            lut.put(0, 0, data)
            Core.LUT(rgb, lut, rgb)
            lut.release()
        }

        val den = Mat(); Imgproc.bilateralFilter(rgb, den, 5, 25.0, 5.0); rgb.release(); rgb = den

        run {                                                            // edge-aware unsharp mask
            val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
            val edges = Mat(); Imgproc.Canny(gray, edges, 60.0, 150.0)
            Imgproc.dilate(edges, edges, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)))
            val maskF = Mat(); edges.convertTo(maskF, CvType.CV_32F, 1.0 / 255.0)
            Imgproc.GaussianBlur(maskF, maskF, Size(0.0, 0.0), 2.0)
            val mask3 = Mat(); Core.merge(listOf(maskF, maskF, maskF), mask3)
            val f = Mat(); rgb.convertTo(f, CvType.CV_32FC3)
            val blur = Mat(); Imgproc.GaussianBlur(f, blur, Size(0.0, 0.0), 1.2)
            val detail = Mat(); Core.subtract(f, blur, detail)
            Core.multiply(detail, mask3, detail)
            val amount = if ("blurry" in flags) 1.2 else 0.6
            val out = Mat(); Core.addWeighted(f, 1.0, detail, amount, 0.0, out)
            out.convertTo(rgb, CvType.CV_8UC3)
            listOf(gray, edges, maskF, mask3, f, blur, detail, out).forEach { it.release() }
        }

        if (target != null && target != rgb.cols()) {                    // resolution optimisation: generate small, upscale
            val up = Mat(); Imgproc.resize(rgb, up, Size(target.toDouble(), target.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)
            rgb.release(); rgb = up
        }
        val after = assess(rgb)
        val outBmp = toBitmap(rgb)
        rgb.release()
        return Triple(outBmp, before, after)
    }
}
