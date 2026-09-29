package com.fanlaoshi.novascan

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var imageCapture: ImageCapture? = null
    private var previewView: PreviewView? = null
    private val detector = DocumentDetector()
    private val capturing = AtomicBoolean(false)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) setContent { ScannerScreen() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setContent { ScannerScreen() }
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    private fun startCamera(onDetection: (DocumentDetector.Detection?) -> Unit) {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build()
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            imageCapture = capture
            previewView?.let { preview.setSurfaceProvider(it.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { proxy ->
                try {
                    val y = proxy.planes.firstOrNull()?.buffer ?: return@setAnalyzer
                    val bytes = ByteArray(y.remaining())
                    y.get(bytes)
                    val mat = Mat(proxy.height, proxy.width, org.opencv.core.CvType.CV_8UC1)
                    mat.put(0, 0, bytes)
                    val detection = detector.detect(mat)
                    runOnUiThread { onDetection(detection) }
                    mat.release()
                } catch (_: Exception) {
                    runOnUiThread { onDetection(null) }
                } finally {
                    proxy.close()
                }
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture, analysis)
            } catch (_: Exception) {
                // Camera binding failure is surfaced by the platform; keep UI alive.
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.compose.runtime.Composable
    private fun ScannerScreen() {
        var detection by remember { mutableStateOf<DocumentDetector.Detection?>(null) }
        var ready by remember { mutableStateOf(false) }
        val context = LocalContext.current

        androidx.compose.runtime.LaunchedEffect(Unit) {
            startCamera { d ->
                detection = d
                ready = d != null && d.confidence > 0.35
            }
        }

        Box(Modifier.fillMaxSize().background(Color.Black)) {
            AndroidView(
                factory = {
                    PreviewView(context).also {
                        it.scaleType = PreviewView.ScaleType.FILL_CENTER
                        previewView = it
                    }
                },
                modifier = Modifier.fillMaxSize()
            )

            detection?.let { d ->
                Canvas(Modifier.fillMaxSize()) {
                    val sx = size.width / d.width.toFloat()
                    val sy = size.height / d.height.toFloat()
                    val path = Path()
                    d.points.forEachIndexed { index, p ->
                        val x = p.x.toFloat() * sx
                        val y = p.y.toFloat() * sy
                        if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
                    }
                    path.close()
                    drawPath(path, Color(0xFF49E66B), style = androidx.compose.ui.graphics.drawscope.Stroke(width = 5.dp.toPx()))
                }
            }

            Column(
                Modifier.fillMaxWidth().align(Alignment.TopCenter).padding(top = 42.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("NovaScan", color = Color.White, fontSize = 20.sp)
                Text(
                    if (ready) "已检测到纸张" else "请将纸张放入画面",
                    color = if (ready) Color(0xFF49E66B) else Color.White,
                    fontSize = 14.sp,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }

            Row(
                Modifier.fillMaxWidth().align(Alignment.BottomCenter).padding(bottom = 42.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = { takePhoto() },
                    modifier = Modifier.size(76.dp)
                ) {
                    Canvas(Modifier.size(70.dp)) {
                        drawCircle(Color.White)
                        drawCircle(Color.Black, radius = 27.dp.toPx())
                        drawCircle(Color.White, radius = 22.dp.toPx())
                    }
                }
            }
        }
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        if (!capturing.compareAndSet(false, true)) return
        val file = java.io.File(cacheDir, "scan_${SystemClock.elapsedRealtime()}.jpg")
        val options = ImageCapture.OutputFileOptions.Builder(file).build()
        capture.takePicture(options, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
            override fun onError(exception: ImageCaptureException) {
                capturing.set(false)
            }
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                capturing.set(false)
                // V1 keeps the captured original. Perspective correction/export is the next layer.
            }
        })
    }
}
