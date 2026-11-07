package com.example.t2imobile

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import java.io.File

enum class Budget(val label: String, val seconds: Double) {
    AUTO("Auto (device-aware)", 30.0),
    FAST("Fast (<= 10 s)", 10.0),
    BALANCED("Balanced (<= 25 s)", 25.0),
    QUALITY("Quality (<= 60 s)", 60.0),
    ULTRA("Ultra Quality (no limit)", Double.MAX_VALUE)   // ← new: no time cap, best possible result
}

data class Plan(
    val size: Int,
    val steps: Int,
    val lowMemory: Boolean,
    val index: Int,
    val reason: String,
    val guidanceScale: Float = 0f,   // 0 = LCM mode (no CFG); >1 enables classifier-free guidance
    val negativePrompt: String = ""  // only used when guidanceScale > 0
)

/**
 * Adaptive, resource-aware generation planner.
 *
 * Ladders:
 *  - Standard ladder (AUTO / FAST / BALANCED / QUALITY): LCM mode, 1–8 steps.
 *    Ordered best→cheapest so the first rung that fits the time + RAM budget is used.
 *  - FP32 runs use low-strength classifier-free guidance; other variants use plain LCM.
 *  - Ultra ladder: LCM at higher step counts + 512 px.
 *    No time limit, so it always picks the highest quality rung that fits in RAM.
 *
 * downgrade() gives the next cheaper rung for OOM / session-error fallback.
 */
class BudgetPlanner(private val ctx: Context) {

    // ── standard quality ladder (LCM) ─────────────────────────────────────
    private val standardLadder = listOf(
        Triple(512, 8, 0f),   // best standard tier
        Triple(512, 4, 0f),
        Triple(512, 3, 0f),
        Triple(384, 8, 0f),
        Triple(384, 4, 0f),
        Triple(384, 3, 0f),
        Triple(512, 2, 0f),
        Triple(384, 2, 0f),
        Triple(256, 4, 0f),
        Triple(256, 2, 0f),
        Triple(256, 1, 0f)    // cheapest fallback
    )

    // ── ultra quality ladder (no time limit) ──────────────────────────────
    // All entries are 512 px; steps go high because there is no latency cap.
    // Low-strength guidance is enabled only for the fp32 variant.
    private val ultraLadder = listOf(
        Triple(512, 8, 0f),
        Triple(512, 6, 0f),
        Triple(512, 4, 0f)    // minimum ultra rung
    )

    private val prefs = ctx.getSharedPreferences("budget", Context.MODE_PRIVATE)

    // Ignore timing estimates learned with the previous standard ladder.
    private fun key(variant: String) = "ms_step_512_v2_$variant"

    /** ms per UNet step at 512 px; updated via EMA after each real run. */
    private fun msPerStep512(variant: String): Double =
        prefs.getFloat(key(variant), if (variant == "fp32") 9000f else 5000f).toDouble()

    fun recordRun(variant: String, size: Int, steps: Int, guidanceScale: Float, unetMs: Long) {
        if (steps <= 0) return
        val passes = if (guidanceScale > 1f) 2 else 1
        val norm = unetMs.toDouble() / steps / passes / ((size / 512.0) * (size / 512.0))
        val prev = msPerStep512(variant)
        val hasPrev = prefs.contains(key(variant))
        prefs.edit()
            .putFloat(key(variant), (if (hasPrev) 0.6 * prev + 0.4 * norm else norm).toFloat())
            .apply()
    }

    fun availMemMb(): Long {
        val mi = ActivityManager.MemoryInfo()
        (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        return mi.availMem / (1024 * 1024)
    }

    private fun unetFileMb(root: File, variant: String): Long =
        File(root, variant).listFiles { f -> f.name.startsWith("unet.onnx") }
            ?.sumOf { it.length() }?.div(1024 * 1024) ?: 1700

    private fun thermalSevere(): Boolean {
        if (Build.VERSION.SDK_INT < 29) return false
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE
    }

    private fun powerSave(): Boolean =
        (ctx.getSystemService(Context.POWER_SERVICE) as PowerManager).isPowerSaveMode

    private fun estimateMemMb(unetMb: Long, size: Int, lowMem: Boolean): Long {
        val act = (700 * (size / 512.0) * (size / 512.0)).toLong()
        return if (lowMem) unetMb + act else (unetMb * 1.25).toLong() + 600 + act
    }

    fun plan(root: File, variant: String, budget: Budget): Plan {
        val avail   = availMemMb()
        val unetMb  = unetFileMb(root, variant)
        val hot     = thermalSevere() || powerSave()
        val ms512   = msPerStep512(variant)
        val guidanceScale = if (variant == "fp32") 1.5f else 0f
        val unetPasses = if (guidanceScale > 1f) 2 else 1

        // ── Ultra: ignore thermal/battery warnings, pick highest-quality RAM-safe rung ──
        if (budget == Budget.ULTRA) {
            for ((idx, r) in ultraLadder.withIndex()) {
                val (size, steps, _) = r
                val lowMem = avail < estimateMemMb(unetMb, size, false)
                if (avail >= estimateMemMb(unetMb, size, lowMem)) {
                    return Plan(
                        size, steps, lowMem, idx,
                        "ULTRA: avail=${avail}MB, ${steps} steps @ ${size}px, no time limit",
                        guidanceScale = guidanceScale
                    )
                }
            }
            // If even the smallest ultra rung doesn't fit memory, fall back to best standard rung
            return Plan(512, 4, true, 0,
                "ULTRA: low memory fallback (avail=${avail}MB)", guidanceScale = guidanceScale)
        }

        // ── Standard modes: pick first rung within time + RAM budget ──────
        val seconds = when (budget) {
            Budget.AUTO      -> if (hot) 12.0 else 30.0
            Budget.FAST      -> 10.0
            Budget.BALANCED  -> 25.0
            Budget.QUALITY   -> 60.0
            Budget.ULTRA     -> Double.MAX_VALUE  // handled above, never reached
        }

        for ((idx, r) in standardLadder.withIndex()) {
            val (size, steps, _) = r
            val est    = (ms512 * steps * unetPasses *
                (size / 512.0) * (size / 512.0) + 1500) / 1000.0
            val lowMem = avail < estimateMemMb(unetMb, size, false)
            if (est <= seconds && avail >= estimateMemMb(unetMb, size, lowMem)) {
                return Plan(
                    size, steps, lowMem, idx,
                    "avail=${avail}MB, est=%.1fs <= %.0fs%s%s".format(
                        est, seconds,
                        if (lowMem) ", low-memory mode" else "",
                        if (hot) ", thermal/battery-saver" else ""
                    ),
                    guidanceScale = guidanceScale
                )
            }
        }

        // Last-resort minimum
        return Plan(256, 1, true, standardLadder.lastIndex,
            "minimum tier (avail=${avail}MB)", guidanceScale = guidanceScale)
    }

    fun downgrade(p: Plan): Plan? {
        val ladder = standardLadder   // Ultra failures always fall back to standard ladder
        val i = p.index + 1
        if (i >= ladder.size) return null
        val (s, st, _) = ladder[i]
        return Plan(s, st, true, i, "fallback after failure", guidanceScale = p.guidanceScale)
    }
}
