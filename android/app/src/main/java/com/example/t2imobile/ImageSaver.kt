package com.example.t2imobile

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

/**
 * Saves a generated image to the phone's Gallery (Pictures/T2IMobile/) and
 * provides a content:// URI for the Android share sheet.
 *
 * Two entry points:
 *   saveToGallery()  — writes to MediaStore (visible in Photos / Gallery app)
 *   shareImage()     — opens the system share sheet so the user can send to
 *                      WhatsApp, Telegram, Gmail, Drive, etc.
 *
 * Both work without requesting WRITE_EXTERNAL_STORAGE on Android 10+ (API 29+).
 * On Android 9 and below the manifest already declares the permission with maxSdkVersion=28.
 */
object ImageSaver {

    private const val AUTHORITY = "com.example.t2imobile.fileprovider"
    private const val ALBUM     = "T2IMobile"

    /**
     * Saves [bitmap] as a PNG in the phone's gallery.
     * Returns the URI of the saved image, or null if saving failed.
     * Shows a Toast on success or failure.
     */
    fun saveToGallery(ctx: Context, bitmap: Bitmap, fileName: String = makeFileName()): Uri? {
        return try {
            val uri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android 10+ — use MediaStore, no storage permission needed
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH,
                        "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val resolver = ctx.contentResolver
                val imageUri = resolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
                ) ?: throw Exception("MediaStore insert returned null")

                resolver.openOutputStream(imageUri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }

                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(imageUri, values, null, null)
                imageUri
            } else {
                // Android 9 and below — write to public Pictures directory directly
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_PICTURES
                    ), ALBUM
                ).apply { mkdirs() }
                val file = File(dir, fileName)
                FileOutputStream(file).use { out: OutputStream ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
                // Tell MediaStore the file exists so Gallery picks it up
                ctx.sendBroadcast(
                    Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, Uri.fromFile(file))
                )
                Uri.fromFile(file)
            }

            Toast.makeText(ctx, "Saved to Gallery ✓", Toast.LENGTH_SHORT).show()
            uri
        } catch (e: Exception) {
            Toast.makeText(ctx, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
            null
        }
    }

    /**
     * Saves [bitmap] to the app's private outputs/ folder (same place MainActivity
     * already saves PNGs), then opens the system share sheet via a FileProvider URI.
     * The user can send the image to any installed app.
     */
    fun shareImage(ctx: Context, bitmap: Bitmap, fileName: String = makeFileName()) {
        try {
            // Write to the app's own outputs/ folder (no permission needed)
            val outDir = File(ctx.getExternalFilesDir(null), "outputs").apply { mkdirs() }
            val file   = File(outDir, fileName)
            FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }

            // Convert to a content:// URI that other apps can read
            val uri = FileProvider.getUriForFile(ctx, AUTHORITY, file)

            val intent = Intent(Intent.ACTION_SEND).apply {
                type    = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "Generated image — T2I Mobile")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            ctx.startActivity(Intent.createChooser(intent, "Share image via…"))
        } catch (e: Exception) {
            Toast.makeText(ctx, "Share failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun makeFileName(): String =
        "t2i_${System.currentTimeMillis()}.png"
}
