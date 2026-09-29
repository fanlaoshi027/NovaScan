package com.fanlaoshi.novascan

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot

/** Lightweight real-time detector. It favors stable, large quadrilaterals over noisy edges. */
class DocumentDetector {
    data class Detection(
        val points: List<Point>,
        val confidence: Double,
        val width: Int,
        val height: Int
    )

    private var previous: List<Point>? = null
    private var stableFrames = 0

    fun detect(gray: Mat): Detection? {
        if (gray.empty()) return null

        val work = Mat()
        val blurred = Mat()
        val edges = Mat()
        try {
            val scale = if (gray.width() > 900) 900.0 / gray.width() else 1.0
            if (scale < 1.0) {
                Imgproc.resize(gray, work, Size(), scale, scale, Imgproc.INTER_AREA)
            } else {
                gray.copyTo(work)
            }

            Imgproc.GaussianBlur(work, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 55.0, 160.0)

            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(edges, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

            val imageArea = work.width().toDouble() * work.height().toDouble()
            var best: List<Point>? = null
            var bestScore = 0.0

            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area < imageArea * 0.18 || area > imageArea * 0.98) {
                    contour.release()
                    continue
                }

                val curve = MatOfPoint2f(*contour.toArray())
                val perimeter = Imgproc.arcLength(curve, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(curve, approx, perimeter * 0.025, true)
                val pts = approx.toArray()

                if (pts.size == 4 && Imgproc.isContourConvex(MatOfPoint(*pts))) {
                    val rectangularity = area / (imageArea)
                    val score = rectangularity + 0.15 * shapeQuality(pts)
                    if (score > bestScore) {
                        bestScore = score
                        best = orderPoints(pts).map { Point(it.x / scale, it.y / scale) }
                    }
                }
                curve.release()
                approx.release()
                contour.release()
            }

            val result = best ?: run {
                previous = null
                stableFrames = 0
                return null
            }

            val stable = previous?.let { sameShape(it, result, gray.width(), gray.height()) } ?: false
            stableFrames = if (stable) stableFrames + 1 else 1
            previous = result

            return Detection(result, (bestScore * 1.7).coerceIn(0.0, 1.0), gray.width(), gray.height())
        } finally {
            work.release()
            blurred.release()
            edges.release()
        }
    }

    fun isStable(minFrames: Int = 8): Boolean = stableFrames >= minFrames

    private fun shapeQuality(p: List<Point>): Double {
        val sides = (0 until 4).map { distance(p[it], p[(it + 1) % 4]) }
        val max = sides.maxOrNull() ?: return 0.0
        val min = sides.minOrNull() ?: return 0.0
        return if (max <= 0.0) 0.0 else min / max
    }

    private fun distance(a: Point, b: Point): Double = hypot(a.x - b.x, a.y - b.y)

    private fun sameShape(a: List<Point>, b: List<Point>, w: Int, h: Int): Boolean {
        val tolerance = 0.045 * hypot(w.toDouble(), h.toDouble())
        return a.zip(b).all { distance(it.first, it.second) < tolerance }
    }

    private fun orderPoints(points: Array<Point>): Array<Point> {
        val sum = points.map { it.x + it.y }
        val diff = points.map { it.x - it.y }
        return arrayOf(
            points[sum.indexOf(sum.minOrNull()!!)],
            points[diff.indexOf(diff.maxOrNull()!!)],
            points[sum.indexOf(sum.maxOrNull()!!)],
            points[diff.indexOf(diff.minOrNull()!!)]
        )
    }
}
