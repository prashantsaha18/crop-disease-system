// classifier.dart
// ────────────────────────────────────────────────────────────────────────────
// Handles:
//   1. Loading crop_model.tflite from Flutter assets.
//   2. Pre-processing the input image (resize → normalize).
//   3. Running inference via tflite_flutter.
//   4. Computing a client-side GradCAM-style heatmap overlay
//      (approximated via saliency mapping since GradCAM requires model internals
//      not exposed through TFLite; for production, embed the overlay generation
//      in Python and bundle pre-computed overlays, or use the full Keras model).
// ────────────────────────────────────────────────────────────────────────────

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

// ─────────────────────────── constants ────────────────────────────────────── //

const int _inputSize  = 224;
const int _numClasses = 38;

// EfficientNetV2 preprocess_input: scale to [-1, 1]
const double _mean = 127.5;
const double _std  = 127.5;

// ─────────────────────────── data classes ─────────────────────────────────── //

/// Full result returned by [CropClassifier.classify].
class ClassificationResult {
  final String  label;           // e.g. "Apple___Apple_scab"
  final String  displayName;     // human-friendly: "Apple – Apple Scab"
  final double  confidence;      // 0.0 – 1.0
  final List<TopPrediction> top5;
  final img.Image? heatmapOverlay; // GradCAM-style overlay (nullable until computed)
  final Duration inferenceTime;

  const ClassificationResult({
    required this.label,
    required this.displayName,
    required this.confidence,
    required this.top5,
    required this.inferenceTime,
    this.heatmapOverlay,
  });
}

class TopPrediction {
  final String label;
  final String displayName;
  final double confidence;
  const TopPrediction({
    required this.label,
    required this.displayName,
    required this.confidence,
  });
}

// ─────────────────────────── treatment DB ─────────────────────────────────── //

/// Minimal offline treatment recommendations keyed by label.
/// In production, ship a full JSON asset.
const Map<String, String> kTreatmentDB = {
  'Apple___Apple_scab':
      'Apply fungicide (e.g. captan or mancozeb) at bud break. '
      'Remove infected leaves and fruit. '
      'Ensure good air circulation through pruning.',
  'Apple___Black_rot':
      'Prune and destroy infected branches. Apply copper-based fungicide. '
      'Avoid overhead irrigation.',
  'Apple___Cedar_apple_rust':
      'Apply myclobutanil or propiconazole at pink bud stage. '
      'Remove nearby juniper/cedar hosts if possible.',
  'Apple___healthy':
      'No disease detected. Maintain regular fertilisation and irrigation schedules.',
  'Blueberry___healthy':
      'No disease detected. Monitor soil pH (4.5–5.5) and water consistently.',
  'Cherry_(including_sour)___Powdery_mildew':
      'Apply sulfur-based or potassium bicarbonate fungicide. '
      'Increase air circulation; avoid wetting foliage.',
  'Cherry_(including_sour)___healthy':
      'No disease detected. Keep area weed-free and irrigate at ground level.',
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
      'Rotate crops annually. Apply strobilurin fungicide at silking stage. '
      'Use resistant hybrids.',
  'Corn_(maize)___Common_rust_':
      'Apply triazole or strobilurin fungicide early. Plant rust-resistant varieties.',
  'Corn_(maize)___Northern_Leaf_Blight':
      'Apply fungicide (propiconazole) at VT/R1 stage. '
      'Rotate with non-host crops. Use resistant varieties.',
  'Corn_(maize)___healthy':
      'No disease detected. Follow standard integrated pest management practices.',
  'Grape___Black_rot':
      'Apply captan or mancozeb on a 7-10 day schedule. '
      'Remove mummified berries. Prune for air circulation.',
  'Grape___Esca_(Black_Measles)':
      'Remove and destroy infected wood. Apply wound sealant after pruning. '
      'There is no curative treatment; manage with preventive pruning hygiene.',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
      'Spray copper-based fungicide. Improve canopy management. '
      'Collect and destroy fallen leaves.',
  'Grape___healthy':
      'No disease detected. Maintain balanced fertilisation and regular monitoring.',
  'Orange___Haunglongbing_(Citrus_greening)':
      'Remove and destroy infected trees. Control Asian citrus psyllid vector '
      'with imidacloprid. There is currently no cure — prevention is critical.',
  'Peach___Bacterial_spot':
      'Apply copper bactericide early in the season. '
      'Avoid overhead sprinklers. Use resistant varieties.',
  'Peach___healthy':
      'No disease detected. Monitor for peach leaf curl in spring.',
  'Pepper,_bell___Bacterial_spot':
      'Use copper-based bactericide. Rotate crops. '
      'Avoid working in fields when wet.',
  'Pepper,_bell___healthy':
      'No disease detected. Ensure adequate calcium nutrition to prevent blossom end rot.',
  'Potato___Early_blight':
      'Apply chlorothalonil or mancozeb every 7-10 days. '
      'Hill soil around plants. Remove infected lower leaves.',
  'Potato___Late_blight':
      'Apply metalaxyl + mancozeb immediately. Destroy infected plants. '
      'Avoid excessive nitrogen. Monitor weather for infection windows.',
  'Potato___healthy':
      'No disease detected. Monitor for Colorado potato beetle and aphids.',
  'Raspberry___healthy':
      'No disease detected. Prune out old floricanes after harvest.',
  'Soybean___healthy':
      'No disease detected. Scout regularly for sudden death syndrome and SCN.',
  'Squash___Powdery_mildew':
      'Apply potassium bicarbonate or sulfur spray. '
      'Avoid overhead watering. Remove severely infected leaves.',
  'Strawberry___Leaf_scorch':
      'Apply captan fungicide. Remove infected leaves. '
      'Ensure good drainage and air circulation.',
  'Strawberry___healthy':
      'No disease detected. Replace planting every 3-4 years.',
  'Tomato___Bacterial_spot':
      'Use copper + mancozeb combination. Rotate crops. '
      'Avoid working among wet plants.',
  'Tomato___Early_blight':
      'Apply chlorothalonil or azoxystrobin. Stake plants for airflow. '
      'Mulch to prevent soil splash.',
  'Tomato___Late_blight':
      'Apply metalaxyl-based fungicide immediately. Remove infected tissue. '
      'Never compost late blight material.',
  'Tomato___Leaf_Mold':
      'Reduce humidity; improve ventilation in greenhouses. '
      'Apply chlorothalonil or copper fungicide.',
  'Tomato___Septoria_leaf_spot':
      'Remove infected leaves. Apply mancozeb or chlorothalonil every 7-10 days.',
  'Tomato___Spider_mites Two-spotted_spider_mite':
      'Apply miticide (abamectin or spinosad). '
      'Increase humidity. Introduce predatory mites (Phytoseiulus persimilis).',
  'Tomato___Target_Spot':
      'Apply azoxystrobin or tebuconazole fungicide. '
      'Rotate crops and remove crop debris.',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
      'Control whitefly vector with insecticidal soap or neonicotinoids. '
      'Remove infected plants. Use reflective mulches.',
  'Tomato___Tomato_mosaic_virus':
      'Remove and destroy infected plants. '
      'Disinfect tools with bleach solution. Control aphid vectors.',
  'Tomato___healthy':
      'No disease detected. Maintain consistent irrigation and calcium nutrition.',
};

/// Convert a raw class label to a human-readable display name.
String labelToDisplayName(String label) {
  // e.g.  "Tomato___Early_blight"  →  "Tomato – Early Blight"
  //        "Apple___healthy"       →  "Apple – Healthy"
  final parts = label.split('___');
  if (parts.length < 2) return label;
  final plant   = parts[0].replaceAll('_', ' ');
  final disease = parts[1].replaceAll('_', ' ').split(' ')
      .map((w) => w.isEmpty ? '' : '${w[0].toUpperCase()}${w.substring(1)}')
      .join(' ');
  return '$plant – $disease';
}

// ─────────────────────────── classifier ───────────────────────────────────── //

class CropClassifier {
  CropClassifier._();

  static CropClassifier? _instance;
  static CropClassifier get instance => _instance!;

  late final Interpreter    _interpreter;
  late final List<String>   _labels;
  bool _loaded = false;

  /// Load the TFLite model and labels from Flutter assets.
  static Future<CropClassifier> load() async {
    if (_instance != null && _instance!._loaded) return _instance!;

    final clf = CropClassifier._();

    // ── model ────────────────────────────────────────────────────────────── //
    final interpreterOptions = InterpreterOptions()
      ..threads = 4;
    clf._interpreter = await Interpreter.fromAsset(
      'assets/models/crop_model.tflite',
      options: interpreterOptions,
    );

    // ── labels ───────────────────────────────────────────────────────────── //
    final labelsStr = await rootBundle.loadString('assets/models/labels.txt');
    clf._labels = labelsStr
        .trim()
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();

    assert(
      clf._labels.length == _numClasses,
      'Expected $_numClasses labels, got ${clf._labels.length}',
    );

    clf._loaded = true;
    _instance  = clf;
    return clf;
  }

  void dispose() {
    _interpreter.close();
    _loaded   = false;
    _instance = null;
  }

  // ── public API ────────────────────────────────────────────────────────── //

  /// Classify [imageFile] and return a [ClassificationResult].
  Future<ClassificationResult> classify(File imageFile) async {
    final bytes    = await imageFile.readAsBytes();
    final original = img.decodeImage(bytes);
    if (original == null) throw Exception('Could not decode image');

    final resized    = img.copyResize(original, width: _inputSize, height: _inputSize);
    final inputTensor = _imageToFloat32(resized);
    final output      = List.generate(1, (_) => List<double>.filled(_numClasses, 0));

    final sw = Stopwatch()..start();
    _interpreter.run(inputTensor, output);
    sw.stop();

    final scores = output[0];
    final maxIdx = _argmax(scores);
    final conf   = scores[maxIdx];

    // top-5
    final indexed = scores.asMap().entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final top5 = indexed.take(5).map((e) {
      final lbl = e.key < _labels.length ? _labels[e.key] : 'Unknown';
      return TopPrediction(
        label:       lbl,
        displayName: labelToDisplayName(lbl),
        confidence:  e.value,
      );
    }).toList();

    final label       = maxIdx < _labels.length ? _labels[maxIdx] : 'Unknown';
    final displayName = labelToDisplayName(label);

    // GradCAM-style overlay (approximated on-device)
    final overlay = _buildHeatmapOverlay(original, resized, scores);

    return ClassificationResult(
      label:          label,
      displayName:    displayName,
      confidence:     conf,
      top5:           top5,
      inferenceTime:  sw.elapsed,
      heatmapOverlay: overlay,
    );
  }

  // ── preprocessing ─────────────────────────────────────────────────────── //

  /// Convert a resized [img.Image] to a (1, 224, 224, 3) Float32List.
  List<List<List<List<double>>>> _imageToFloat32(img.Image src) {
    return List.generate(1, (_) =>
      List.generate(_inputSize, (y) =>
        List.generate(_inputSize, (x) {
          final pixel = src.getPixel(x, y);
          return [
            (pixel.r.toDouble() - _mean) / _std,
            (pixel.g.toDouble() - _mean) / _std,
            (pixel.b.toDouble() - _mean) / _std,
          ];
        }),
      ),
    );
  }

  // ── approximate heatmap overlay ───────────────────────────────────────── //

  /// Builds a JET-coloured saliency overlay approximating GradCAM visually.
  ///
  /// Since TFLite doesn't expose intermediate activations, we compute an
  /// approximate spatial importance map by:
  ///   1. Dividing the 224×224 input into 7×7 patches.
  ///   2. Zeroing each patch and re-running inference.
  ///   3. Measuring the drop in top-class confidence → importance score.
  ///   4. Upsampling and JET-colouring the 7×7 importance map.
  ///   5. Alpha-blending with the original image.
  ///
  /// This is O(49) extra inferences and takes ~500 ms on a mid-range device.
  /// For production, replace with a server-side or pre-computed GradCAM overlay.
  img.Image? _buildHeatmapOverlay(
    img.Image original,
    img.Image resized,
    List<double> baseScores,
  ) {
    try {
      const gridSize    = 7;
      const patchSize   = _inputSize ~/ gridSize;
      final topClass    = _argmax(baseScores);
      final baseConf    = baseScores[topClass];
      final importance  = List.generate(gridSize, (_) => List<double>.filled(gridSize, 0.0));

      for (int gy = 0; gy < gridSize; gy++) {
        for (int gx = 0; gx < gridSize; gx++) {
          // create a copy with the patch zeroed
          final patched = img.Image.from(resized);
          for (int py = 0; py < patchSize; py++) {
            for (int px = 0; px < patchSize; px++) {
              final x = gx * patchSize + px;
              final y = gy * patchSize + py;
              if (x < _inputSize && y < _inputSize) {
                patched.setPixelRgb(x, y, 0, 0, 0);
              }
            }
          }

          final input  = _imageToFloat32(patched);
          final output = List.generate(1, (_) => List<double>.filled(_numClasses, 0));
          _interpreter.run(input, output);

          // importance = drop in class confidence
          final patchedConf = output[0][topClass];
          importance[gy][gx] = math.max(0, baseConf - patchedConf);
        }
      }

      // ── normalise importance ──────────────────────────────────────────── //
      double maxImp = importance.expand((r) => r).reduce(math.max);
      if (maxImp < 1e-8) maxImp = 1.0;
      final normImp = List.generate(gridSize, (gy) =>
        List.generate(gridSize, (gx) => importance[gy][gx] / maxImp),
      );

      // ── upsample to original resolution ──────────────────────────────── //
      final oW = original.width;
      final oH = original.height;
      final heatmap = img.Image(width: oW, height: oH);

      for (int y = 0; y < oH; y++) {
        for (int x = 0; x < oW; x++) {
          final gx = (x / oW * gridSize).clamp(0, gridSize - 1).toInt();
          final gy = (y / oH * gridSize).clamp(0, gridSize - 1).toInt();
          final imp  = normImp[gy][gx];
          final jet  = _jetColour(imp);
          // blend original + heatmap
          final orig = original.getPixel(x, y);
          final r = (orig.r.toDouble() * 0.6 + jet.$1 * 0.4).round().clamp(0, 255);
          final g = (orig.g.toDouble() * 0.6 + jet.$2 * 0.4).round().clamp(0, 255);
          final b = (orig.b.toDouble() * 0.6 + jet.$3 * 0.4).round().clamp(0, 255);
          heatmap.setPixelRgb(x, y, r, g, b);
        }
      }

      return heatmap;
    } catch (e) {
      // Non-fatal — return null, UI will show original image
      return null;
    }
  }

  // ─────────────────────────── JET colormap ─────────────────────────────── //

  /// Returns (r, g, b) for a value in [0, 1] using the JET colormap.
  static (int, int, int) _jetColour(double v) {
    final r = (255 * _jetSingle(v, 1.5)).clamp(0, 255).round();
    final g = (255 * _jetSingle(v, 1.0)).clamp(0, 255).round();
    final b = (255 * _jetSingle(v, 0.5)).clamp(0, 255).round();
    return (r, g, b);
  }

  static double _jetSingle(double v, double centre) {
    final x = math.max(0.0, 1.5 - (v * 4 - centre * 4).abs());
    return math.min(1.0, x);
  }

  // ─────────────────────────── utilities ────────────────────────────────── //

  static int _argmax(List<double> scores) {
    int best = 0;
    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    return best;
  }

  // ── accessors ─────────────────────────────────────────────────────────── //

  List<String> get labels       => List.unmodifiable(_labels);
  bool         get isLoaded     => _loaded;
}
