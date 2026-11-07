package com.example.t2imobile

import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.FileOutputStream

/**
 * Models are expected in:  <externalFilesDir>/models/   (push with adb, see README)
 *   /sdcard/Android/data/com.example.t2imobile/files/models/{vocab.json,merges.txt,config.json,fp16/,int8/}
 */
class MainActivity : AppCompatActivity() {

    private lateinit var modelsDir: File
    private lateinit var planner: BudgetPlanner
    private var pipeline: T2IPipeline? = null
    private var pipelineKey = ""

    // The last successfully generated (and optionally post-processed) bitmap.
    // Kept in memory so Save and Share don't need to re-read a file.
    private var lastBitmap: Bitmap? = null

    private lateinit var promptEt: EditText
    private lateinit var negativePromptEt: EditText
    private lateinit var budgetSp: Spinner
    private lateinit var variantSp: Spinner
    private lateinit var providerSp: Spinner
    private lateinit var postCb: CheckBox
    private lateinit var genBtn: Button
    private lateinit var progress: ProgressBar
    private lateinit var imageView: ImageView
    private lateinit var logTv: TextView

    // Save / share UI
    private lateinit var saveShareRow: LinearLayout
    private lateinit var saveBtn: Button
    private lateinit var shareBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        promptEt         = findViewById(R.id.prompt)
        negativePromptEt = findViewById(R.id.negative_prompt)
        budgetSp         = findViewById(R.id.budget)
        variantSp        = findViewById(R.id.variant)
        providerSp       = findViewById(R.id.provider)
        postCb           = findViewById(R.id.post)
        genBtn           = findViewById(R.id.generate)
        progress         = findViewById(R.id.progress)
        imageView        = findViewById(R.id.image)
        logTv            = findViewById(R.id.log)
        saveShareRow     = findViewById(R.id.saveShareRow)
        saveBtn          = findViewById(R.id.saveBtn)
        shareBtn         = findViewById(R.id.shareBtn)

        val cvOk = OpenCVLoader.initLocal()
        modelsDir = File(getExternalFilesDir(null), "models")
        planner   = BudgetPlanner(this)

        budgetSp.adapter  = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            Budget.values().map { it.label })
        providerSp.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            Provider.values().map { it.name })
        val variants = listOf("fp16", "int8", "fp32")
            .filter { File(modelsDir, "$it/unet.onnx").exists() }
        variantSp.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, variants)

        log("OpenCV: ${if (cvOk) "ok" else "FAILED (post-processing disabled)"}")
        log("Models dir: $modelsDir")
        if (variants.isEmpty())
            log("No model variants found. Push deploy/ with adb (see README).")

        genBtn.setOnClickListener { generate(cvOk) }

        // ── Save to Gallery ──────────────────────────────────────────────
        saveBtn.setOnClickListener {
            val bmp = lastBitmap ?: return@setOnClickListener
            ImageSaver.saveToGallery(this, bmp)
        }

        // ── Share sheet ──────────────────────────────────────────────────
        shareBtn.setOnClickListener {
            val bmp = lastBitmap ?: return@setOnClickListener
            ImageSaver.shareImage(this, bmp)
        }
    }

    private fun log(s: String) { logTv.append(s + "\n") }

    private fun generate(cvOk: Boolean) {
        val prompt    = promptEt.text.toString().ifBlank { return }
        val negPrompt = negativePromptEt.text.toString().trim()
        val variant   = variantSp.selectedItem as? String ?: return
        val provider  = Provider.values()[providerSp.selectedItemPosition]
        val budget    = Budget.values()[budgetSp.selectedItemPosition]

        genBtn.isEnabled = false
        progress.visibility = View.VISIBLE
        saveShareRow.visibility = View.GONE   // hide buttons while generating
        logTv.text = ""

        if (budget == Budget.ULTRA) {
            log("Ultra Quality selected — no time limit, using maximum available steps.")
            log("This may take several minutes on-device. Please keep the screen on.")
        }

        lifecycleScope.launch {
            try {
                var plan: Plan? = withContext(Dispatchers.Default) {
                    planner.plan(modelsDir, variant, budget)
                }
                log("Plan: ${plan!!.size}px, ${plan.steps} steps, " +
                    "lowMem=${plan.lowMemory}  [${plan.reason}]")
                if (plan.guidanceScale > 1f)
                    log("FP32 guidance enabled (${plan.guidanceScale}x); negative prompt is active.")
                else if (negPrompt.isNotBlank())
                    log("Negative prompt guidance requires the fp32 model; ignored for $variant.")

                var result: GenResult? = null
                while (plan != null && result == null) {
                    val p = plan
                    try {
                        result = withContext(Dispatchers.Default) {
                            run(prompt, negPrompt, variant, provider, p)
                        }
                    } catch (e: Throwable) {
                        log("FAILED at ${p.size}px/${p.steps} steps: " +
                            "${e.javaClass.simpleName}: ${e.message?.take(120)}")
                        pipeline?.close(); pipeline = null; pipelineKey = ""
                        plan = planner.downgrade(p)
                        if (plan != null)
                            log("Fallback -> ${plan.size}px, ${plan.steps} steps, low-memory mode")
                    }
                }

                if (result == null) {
                    log("Could not generate on this device with the available models.")
                    return@launch
                }

                planner.recordRun(
                    variant, result.size, result.steps, plan?.guidanceScale ?: 0f, result.timings.unetMs
                )
                val t = result.timings
                log("Timings ms: load=${t.loadMs} tok=${t.tokenizeMs} " +
                    "text=${t.textMs} unet=${t.unetMs} " +
                    "(${t.unetMs / maxOf(result.steps, 1)}/step) " +
                    "vae=${t.vaeMs} total=${t.totalMs}")
                log("Peak memory: PSS=${t.peakPssMb} MB, " +
                    "native heap=${t.peakNativeMb} MB   ${t.providerNote}")

                var shown: Bitmap = result.bitmap
                if (postCb.isChecked && cvOk) {
                    val (bmp, before, after) = withContext(Dispatchers.Default) {
                        PostProcess.enhance(result.bitmap, 512)
                    }
                    log("IQA before: $before")
                    log("IQA after : $after")
                    shown = bmp
                }

                // Store the final bitmap so Save/Share buttons can use it
                lastBitmap = shown
                imageView.setImageBitmap(shown)

                // Reveal the Save / Share buttons now that there is an image
                saveShareRow.visibility = View.VISIBLE

                // Still write the internal bench log as before
                saveAndLog(shown, prompt, negPrompt, variant, provider, budget, result)

            } finally {
                progress.visibility = View.INVISIBLE
                genBtn.isEnabled = true
            }
        }
    }

    private fun run(
        prompt: String,
        negativePrompt: String,
        variant: String,
        provider: Provider,
        plan: Plan
    ): GenResult {
        val key = "$variant/${provider.name}/${plan.lowMemory}"
        if (pipeline == null || pipelineKey != key) {
            pipeline?.close()
            pipeline = T2IPipeline(modelsDir, variant, provider, plan.lowMemory)
            pipelineKey = key
        }
        return pipeline!!.generate(
            prompt         = prompt,
            steps          = plan.steps,
            size           = plan.size,
            seed           = System.nanoTime() and 0xFFFFFF,
            negativePrompt = negativePrompt.ifBlank { null },
            guidanceScale  = plan.guidanceScale,
            onStage        = { stage -> runOnUiThread { genBtn.text = stage } }
        ).also { runOnUiThread { genBtn.text = "Generate" } }
    }

    private fun saveAndLog(
        bmp: Bitmap,
        prompt: String,
        negPrompt: String,
        variant: String,
        provider: Provider,
        budget: Budget,
        r: GenResult
    ) {
        // Internal PNG copy in app's private folder (for bench_log traceability)
        val out = File(getExternalFilesDir(null), "outputs").apply { mkdirs() }
        val f   = File(out, "img_${System.currentTimeMillis()}.png")
        FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }

        val csv = File(getExternalFilesDir(null), "bench_log.csv")
        if (!csv.exists())
            csv.writeText(
                "time,variant,provider,budget,size,steps,load_ms,text_ms," +
                "unet_ms,vae_ms,total_ms,peak_pss_mb,peak_native_mb,prompt,neg_prompt\n"
            )
        val t = r.timings
        csv.appendText(
            "${System.currentTimeMillis()},$variant,${provider.name},${budget.name}," +
            "${r.size},${r.steps},${t.loadMs},${t.textMs},${t.unetMs},${t.vaeMs}," +
            "${t.totalMs},${t.peakPssMb},${t.peakNativeMb}," +
            "\"${prompt.replace("\"", "'")}\",\"${negPrompt.replace("\"", "'")}\"\n"
        )
        log("Saved internally: ${f.name}")
        log("Tap ⬇ Save to Gallery to add it to your Photos app.")
    }

    override fun onDestroy() { pipeline?.close(); super.onDestroy() }
}
