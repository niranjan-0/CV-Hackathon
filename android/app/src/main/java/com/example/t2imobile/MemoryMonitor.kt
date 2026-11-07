package com.example.t2imobile

import android.os.Debug

/** Samples process memory every 100 ms while a generation runs. */
class MemoryMonitor {
    @Volatile var peakPssMb = 0L
    @Volatile var peakNativeMb = 0L
    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        running = true
        thread = Thread {
            val mi = Debug.MemoryInfo()
            while (running) {
                sample(mi)
                try { Thread.sleep(100) } catch (_: InterruptedException) { break }
            }
        }.also { it.isDaemon = true; it.start() }
    }

    private fun sample(mi: Debug.MemoryInfo) {
        Debug.getMemoryInfo(mi)
        peakPssMb = maxOf(peakPssMb, mi.totalPss / 1024L)                 // total PSS (incl. mmap'd model files)
        peakNativeMb = maxOf(peakNativeMb, Debug.getNativeHeapAllocatedSize() / (1024L * 1024L))
    }

    fun stop() {
        running = false
        thread?.join(500)
        sample(Debug.MemoryInfo())
    }
}
