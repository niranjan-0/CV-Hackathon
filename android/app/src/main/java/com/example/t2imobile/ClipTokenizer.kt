package com.example.t2imobile

import org.json.JSONObject

/** CLIP byte-level BPE tokenizer (matches HF CLIPTokenizer for SD 1.5; max length 77). */
class ClipTokenizer(vocabJson: String, mergesTxt: String, private val bos: Int = 49406, private val eos: Int = 49407) {

    private val encoder = HashMap<String, Int>()
    private val ranks = HashMap<String, Int>()          // "a b" -> rank
    private val byteEnc = HashMap<Int, Char>()
    private val cache = HashMap<String, List<String>>()
    private val pat = Regex(
        """<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        RegexOption.IGNORE_CASE
    )

    init {
        val obj = JSONObject(vocabJson)
        for (k in obj.keys()) encoder[k] = obj.getInt(k)
        var r = 0
        mergesTxt.lineSequence().drop(1).forEach { line ->      // first line is "#version"
            if (line.isNotBlank()) ranks[line.trim()] = r++
        }
        // GPT-2 / CLIP bytes_to_unicode
        val bs = ArrayList<Int>()
        bs += ('!'.code..'~'.code); bs += ('¡'.code..'¬'.code); bs += ('®'.code..'ÿ'.code)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) if (b !in bs) { bs += b; cs += 256 + n; n++ }
        for (i in bs.indices) byteEnc[bs[i]] = cs[i].toChar()
    }

    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }
        var word: MutableList<String> = token.map { it.toString() }.toMutableList()
        word[word.size - 1] = word.last() + "</w>"
        while (word.size > 1) {
            var best = Int.MAX_VALUE
            var bestPair: String? = null
            for (i in 0 until word.size - 1) {
                val p = word[i] + " " + word[i + 1]
                val rk = ranks[p]
                if (rk != null && rk < best) { best = rk; bestPair = p }
            }
            if (bestPair == null) break
            val (a, b) = bestPair.split(" ")
            val merged = ArrayList<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == a && word[i + 1] == b) { merged += a + b; i += 2 }
                else { merged += word[i]; i++ }
            }
            word = merged
        }
        cache[token] = word
        return word
    }

    /** Returns exactly [maxLen] ids: BOS + tokens + EOS, padded with EOS (SD 1.5 pad token). */
    fun encode(text: String, maxLen: Int = 77): LongArray {
        val clean = text.replace(Regex("\\s+"), " ").trim().lowercase()
        val ids = ArrayList<Int>()
        for (m in pat.findAll(clean)) {
            val mapped = m.value.toByteArray(Charsets.UTF_8).joinToString("") { byteEnc[it.toInt() and 0xFF].toString() }
            for (piece in bpe(mapped)) encoder[piece]?.let { ids += it }
        }
        val out = LongArray(maxLen) { eos.toLong() }
        out[0] = bos.toLong()
        val n = minOf(ids.size, maxLen - 2)
        for (i in 0 until n) out[i + 1] = ids[i].toLong()
        out[n + 1] = eos.toLong()
        return out
    }
}
